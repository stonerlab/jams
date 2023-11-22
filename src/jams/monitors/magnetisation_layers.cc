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

  // construct a rotation matrix which will rotate the system so that the norm
  // is always along z
  Mat3 rotation_matrix = rotation_matrix_between_vectors(layer_normal, Vec3{0, 0, 1});

  std::vector<double> rotated_z_position(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r =  rotation_matrix * ::globals::lattice->atom_position(i) * globals::lattice->parameter() * kMeterToNanometer;
    rotated_z_position[i] = r[2];
  }


  // Find the minimum value of z in the rotated system. This will be used as the
  // baseline for layers with a finite thickness.
  double z_min = *std::min_element(rotated_z_position.begin(), rotated_z_position.end());

  // Find the unique layer positions. If the z-component of the position (after
  // rotating the system) is within lattice_tolerance of an existing position
  // we consider them to be in the same layer.
  auto comp_less = [&](double a, double b) -> bool {
    if (layer_thickness == 0.0) {
      return definately_less_than(a, b, jams::defaults::lattice_tolerance);
    }
    return definately_less_than(floor((a - z_min)/layer_thickness) , floor((b - z_min)/layer_thickness), jams::defaults::lattice_tolerance);
  };


  std::map<double, std::vector<int>, decltype(comp_less)> unique_positions(
      comp_less);

  for (auto i = 0; i < globals::num_spins; ++i) {
    unique_positions[rotated_z_position[i]].push_back(i);
  }

  num_layers_ = unique_positions.size();
  layer_spin_indicies_.resize(num_layers_);
  layer_magnetisation_.resize(num_layers_, 3);

  // Move all the data into MultiArrays
  jams::MultiArray<double, 1> layer_positions(num_layers_);
  jams::MultiArray<double, 1> layer_saturation_moment(num_layers_);
  jams::MultiArray<int, 1> layer_spin_count(num_layers_);

  int counter = 0;
  for (auto const &x: unique_positions) {
    layer_positions(counter) = x.first;
    layer_spin_count(counter) = x.second.size(); // number of spins in the layer
    layer_spin_indicies_[counter].resize(x.second.size());

    layer_saturation_moment(counter) = 0.0;
    for (auto i = 0; i < x.second.size(); ++i) {
      auto spin_index = x.second[i];
      layer_spin_indicies_[counter](i) = spin_index;
      layer_saturation_moment(counter) += globals::mus(spin_index) / kBohrMagnetonIU;
    }

    counter++;
  }

  // Create a new h5 file, truncating any old file if it exists.
  HighFive::File file(jams::output::full_path_filename("monitors.h5"),
                      HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

  HighFive::Group group = file.createGroup(h5_group);
  {
    auto dataset = group.createDataSet<int>(
        "num_layers",HighFive::DataSpace::From(num_layers_));
    dataset.write(num_layers_);
  }
  {
    auto dataset = group.createDataSet<double>(
        "layer_normal",HighFive::DataSpace::From(layer_normal));
    dataset.write(layer_normal);
    dataset.createAttribute<std::string>("axis0", "xyz");
  }
  {
    auto dataset = group.createDataSet<double>(
        "layer_thickness",HighFive::DataSpace::From(layer_thickness));
    dataset.write(layer_thickness);
    dataset.createAttribute<std::string>("units", "nm");
    dataset.createAttribute<std::string>("axis0", "layer_index");
    dataset.createAttribute<std::string>("axis1", "layer_thickness");
  }
  {
    auto dataset = group.createDataSet<double>(
        "layer_positions",HighFive::DataSpace::From(layer_positions));
    dataset.write(layer_positions);
    dataset.createAttribute<std::string>("units", "nm");
    dataset.createAttribute<std::string>("axis0", "layer_index");
    dataset.createAttribute<std::string>("axis1", "layer_position");
  }
  {
    auto dataset = group.createDataSet<double>(
        "layer_saturation_moment",HighFive::DataSpace::From(layer_saturation_moment));
    dataset.write(layer_saturation_moment);
    dataset.createAttribute<std::string>("axis0", "layer_index");
    dataset.createAttribute<std::string>("axis1", "magnetisation_xyz");
    dataset.createAttribute<std::string>("units", "bohr_magneton");
  }
  {
    auto dataset = group.createDataSet<int>(
        "layer_spin_count",HighFive::DataSpace::From(layer_spin_count));
    dataset.write(layer_spin_count);
    dataset.createAttribute<std::string>("axis0", "layer_index");
    dataset.createAttribute<std::string>("axis1", "number_of_spins");
  }
}


void MagnetisationLayersMonitor::update(Solver& solver) {

  // Loop over layers and calculate the magnetisation
  for (auto layer_index = 0; layer_index < num_layers_; ++layer_index) {
    Vec3 mag = jams::sum_spins_moments(globals::s, globals::mus,
                                         layer_spin_indicies_[layer_index]);

    // internally we use meV T^-1 for mus so convert back to Bohr magneton
    layer_magnetisation_(layer_index, 0) = mag[0] / kBohrMagnetonIU;
    layer_magnetisation_(layer_index, 1) = mag[1] / kBohrMagnetonIU;
    layer_magnetisation_(layer_index, 2) = mag[2] / kBohrMagnetonIU;
  }

  // Open the h5 file to write new data
  HighFive::File file(
      jams::output::full_path_filename("monitors.h5"), HighFive::File::ReadWrite);

  // Write the data to file
  HighFive::Group group = file.createGroup(
      h5_group + "/timeseries/" + zero_pad_number(solver.iteration(),9));

  group.createAttribute<double>("time", solver.time());
  group.createAttribute<double>("time_step", solver.time_step());
  group.createAttribute<std::string>("units", "ps");

  auto dataset = group.createDataSet<double>(
      "magnetisation",HighFive::DataSpace::From(layer_magnetisation_));
  dataset.createAttribute<std::string>("axis0", "layer_index");
  dataset.createAttribute<std::string>("axis1", "magnetisation_xyz");
  dataset.createAttribute<std::string>("units", "bohr_magneton");

  dataset.write(layer_magnetisation_);
}
