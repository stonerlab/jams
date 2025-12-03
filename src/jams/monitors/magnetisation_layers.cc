// magnetisation_layers.cc                                             -*-C++-*-
#include <jams/monitors/magnetisation_layers.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/cuda/cuda_spin_ops.h>
#include <jams/helpers/maths.h>
#include <jams/helpers/output.h>
#include <jams/helpers/spinops.h>
#include <jams/interface/highfive.h>

#include <map>

MagnetisationLayersMonitor::MagnetisationLayersMonitor(
    const libconfig::Setting &settings)
    : Monitor(settings) {

  Vec3 layer_normal = jams::config_required<Vec3>(settings, "layer_normal");
  auto layer_thickness = jams::config_optional<double>(settings, "layer_thickness", 0.0);
  auto distance_tolerance = jams::config_optional<double>(settings, "distance_tolerance", jams::defaults::lattice_tolerance);

  auto grouping_str = jams::config_optional<std::string>(settings, "grouping", "materials");

  if (lowercase(grouping_str) == "none") {
    grouping_ = Grouping::NONE;
  } else if (lowercase(grouping_str) == "materials") {
    grouping_ = Grouping::MATERIALS;
  } else if (lowercase(grouping_str) == "positions") {
    grouping_ = Grouping::POSITIONS;
  } else {
    throw std::runtime_error("unknown magnetisation grouping: " + grouping_str);
  }

  if (grouping_ == Grouping::NONE) {
    jams::MultiArray<int,1> indices(globals::num_spins);
    for (auto i = 0; i < globals::num_spins; ++i) {
      indices(i) = i;
    }
    group_spin_indices_.push_back(indices);
    group_names_.push_back("total");
  } else if (grouping_ == Grouping::MATERIALS) {
    auto num_groups = globals::lattice->num_materials();
    std::vector<std::vector<int>> material_index_groups(num_groups);
    for (auto i = 0; i < globals::num_spins; ++i) {
      auto group_idx = globals::lattice->lattice_site_material_id(i);
      material_index_groups[group_idx].push_back(i);
    }

    group_spin_indices_.resize(num_groups);
    group_names_.resize(num_groups);
    for (auto group_idx = 0; group_idx < num_groups; ++group_idx) {
      group_spin_indices_[group_idx] = jams::MultiArray<int,1>(material_index_groups[group_idx].begin(), material_index_groups[group_idx].end());
      group_names_[group_idx] = globals::lattice->material_name(group_idx);
    }

  } else if (grouping_ == Grouping::POSITIONS) {
    std::vector<std::vector<int>> position_index_groups(globals::lattice->num_basis_sites());
    for (auto i = 0; i < globals::num_spins; ++i) {
      auto position = globals::lattice->lattice_site_basis_index(i);
      position_index_groups[position].push_back(i);
    }

    group_spin_indices_.resize(position_index_groups.size());
    group_names_.resize(position_index_groups.size());
    for (auto n = 0; n < position_index_groups.size(); ++n) {
      group_spin_indices_[n] = jams::MultiArray<int,1>(position_index_groups[n].begin(), position_index_groups[n].end());
      group_names_[n] = std::to_string(n);
    }
  }

  auto num_groups = group_spin_indices_.size();
  group_num_layers_.resize(num_groups);
  group_layer_spin_indicies_.resize(num_groups);
  group_layer_magnetisation_.resize(num_groups);

  // Create a new h5 file, truncating any old file if it exists.
  HighFive::File file(jams::output::full_path_filename("monitors.h5"),
                      HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

  for (int group_idx = 0; group_idx < group_spin_indices_.size(); ++group_idx) {
    auto num_group_spins = group_spin_indices_[group_idx].size();

    // construct a rotation matrix which will rotate the system so that the norm
    // is always along z
    Mat3 rotation_matrix = rotation_matrix_between_vectors(layer_normal, Vec3{0, 0, 1});

    std::vector<double> rotated_z_position(globals::num_spins);

    // Find the minimum value of z in the rotated system. This will be used as the
    // baseline for layers with a finite thickness.
    double z_min = std::numeric_limits<double>::max();
    for (auto i : group_spin_indices_[group_idx]) {
      auto r = rotation_matrix * ::globals::lattice->lattice_site_position_cart(i)
               * globals::lattice->parameter() * kMeterToNanometer;
      rotated_z_position[i] = r[2];
      if (rotated_z_position[i] < z_min) {
        z_min = rotated_z_position[i];
      }
    }

    // Find the unique layer positions. If the z-component of the position (after
    // rotating the system) is within lattice_tolerance of an existing position
    // we consider them to be in the same layer.
    auto comp_less = [&](double a, double b) -> bool {
      if (layer_thickness == 0.0) {
        return definately_less_than(a, b, distance_tolerance);
      }
      return definately_less_than(floor((a - z_min)/layer_thickness) , floor((b - z_min)/layer_thickness), distance_tolerance);
    };


    std::map<double, std::vector<int>, decltype(comp_less)> unique_positions(
        comp_less);

    for (auto i : group_spin_indices_[group_idx]) {
      unique_positions[rotated_z_position[i]].push_back(i);
    }

    auto num_layers = unique_positions.size();
    group_num_layers_[group_idx] = num_layers;
    group_layer_spin_indicies_[group_idx].resize(num_layers);
    group_layer_magnetisation_[group_idx].resize(num_layers, 3);

    // Move all the data into MultiArrays
    jams::MultiArray<double, 1> layer_positions(num_layers);
    jams::MultiArray<double, 1> layer_saturation_moment(num_layers);
    jams::MultiArray<int, 1> layer_spin_count(num_layers);

    int counter = 0;
    for (auto const &x: unique_positions) {
      layer_positions(counter) = x.first;
      layer_spin_count(counter) = x.second.size(); // number of spins in the layer
      group_layer_spin_indicies_[group_idx][counter].resize(x.second.size());

      layer_saturation_moment(counter) = 0.0;
      for (auto i = 0; i < x.second.size(); ++i) {
        auto spin_index = x.second[i];
        group_layer_spin_indicies_[group_idx][counter](i) = spin_index;
        layer_saturation_moment(counter) += globals::mus(spin_index) / kBohrMagnetonIU;
      }

      counter++;
    }

    HighFive::Group h5_group = file.createGroup(h5_group_root_name_ +"/groups/" + group_names_[group_idx] + "/");
    {
      auto dataset = h5_group.createDataSet<int>(
          "num_layers",HighFive::DataSpace::From(group_num_layers_[group_idx]));
      dataset.write(group_num_layers_[group_idx]);
    }
    {
      auto dataset = h5_group.createDataSet<double>(
          "layer_normal",HighFive::DataSpace::From(layer_normal));
      dataset.write(layer_normal);
      dataset.createAttribute<std::string>("axis0", "xyz");
    }
    {
      auto dataset = h5_group.createDataSet<double>(
          "layer_thickness",HighFive::DataSpace::From(layer_thickness));
      dataset.write(layer_thickness);
      dataset.createAttribute<std::string>("units", "nm");
      dataset.createAttribute<std::string>("axis0", "layer_index");
      dataset.createAttribute<std::string>("axis1", "layer_thickness");
    }
    {
      auto dataset = h5_group.createDataSet<double>(
          "layer_positions",HighFive::DataSpace::From(layer_positions));
      dataset.write(layer_positions);
      dataset.createAttribute<std::string>("units", "nm");
      dataset.createAttribute<std::string>("axis0", "layer_index");
      dataset.createAttribute<std::string>("axis1", "layer_position");
    }
    {
      auto dataset = h5_group.createDataSet<double>(
          "layer_saturation_moment",HighFive::DataSpace::From(layer_saturation_moment));
      dataset.write(layer_saturation_moment);
      dataset.createAttribute<std::string>("axis0", "layer_index");
      dataset.createAttribute<std::string>("axis1", "magnetisation_xyz");
      dataset.createAttribute<std::string>("units", "bohr_magneton");
    }
    {
      auto dataset = h5_group.createDataSet<int>(
          "layer_spin_count",HighFive::DataSpace::From(layer_spin_count));
      dataset.write(layer_spin_count);
      dataset.createAttribute<std::string>("axis0", "layer_index");
      dataset.createAttribute<std::string>("axis1", "number_of_spins");
    }
  }




}


void MagnetisationLayersMonitor::update(Solver& solver) {
  // Open the h5 file to write new data
  HighFive::File file(
      jams::output::full_path_filename("monitors.h5"), HighFive::File::ReadWrite);

  HighFive::Group timeseries_group = file.createGroup(h5_group_root_name_ + "/timeseries/" +  zero_pad_number(solver.iteration(),9));

  timeseries_group.createAttribute<double>("time", solver.time());
  timeseries_group.createAttribute<double>("time_step", solver.time_step());
  timeseries_group.createAttribute<std::string>("units", "ps");

  for (int group_idx = 0; group_idx < group_spin_indices_.size(); ++group_idx) {

    auto spin_group = timeseries_group.createGroup(group_names_[group_idx]);

    // Loop over layers and calculate the magnetisation
    for (auto layer_index = 0; layer_index < group_num_layers_[group_idx]; ++layer_index) {
      Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus,
                                           group_layer_spin_indicies_[group_idx][layer_index]);

      // internally we use meV T^-1 for mus so convert back to Bohr magneton
      group_layer_magnetisation_[group_idx](layer_index, 0) = mag[0] / kBohrMagnetonIU;
      group_layer_magnetisation_[group_idx](layer_index, 1) = mag[1] / kBohrMagnetonIU;
      group_layer_magnetisation_[group_idx](layer_index, 2) = mag[2] / kBohrMagnetonIU;
    }

    auto dataset = spin_group.createDataSet<double>(
        "magnetisation",HighFive::DataSpace::From(group_layer_magnetisation_[group_idx]));
    dataset.createAttribute<std::string>("axis0", "layer_index");
    dataset.createAttribute<std::string>("axis1", "magnetisation_xyz");
    dataset.createAttribute<std::string>("units", "bohr_magneton");

    dataset.write(group_layer_magnetisation_[group_idx]);
  }
}
