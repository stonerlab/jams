// magnetisation_layers.cc                                             -*-C++-*-
#include <jams/monitors/magnetisation_layers.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/cuda/cuda_spin_ops.h>
#include <jams/helpers/maths.h>
#include <jams/helpers/exception.h>
#include <jams/helpers/output.h>
#include <jams/helpers/spinops.h>
#include <jams/interface/highfive.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace {
struct LayerBuildData {
  double position_nm = 0.0;
  std::vector<int> spin_indices;
};

void validate_layer_normal(
    const libconfig::Setting& settings,
    const jams::Vec<double, 3>& layer_normal) {
  for (auto n = 0; n < 3; ++n) {
    if (!std::isfinite(layer_normal[n])) {
      throw jams::ConfigException(settings, "layer_normal components must be finite");
    }
  }

  if (!definately_greater_than(jams::norm(layer_normal), 0.0, std::numeric_limits<double>::epsilon())) {
    throw jams::ConfigException(settings, "layer_normal must not be the zero vector");
  }
}

void validate_non_negative_finite_setting(
    const libconfig::Setting& settings,
    const double value,
    const char* setting_name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw jams::ConfigException(settings, setting_name, " must be finite and non-negative");
  }
}

double projected_layer_position_nm(
    const jams::Vec<double, 3>& layer_normal_unit,
    const int spin_index) {
  // Projecting directly onto the normal avoids building a full rotated
  // coordinate buffer for every spin. The result is the same layer coordinate
  // as the previous rotate-to-z implementation, expressed in nanometres.
  return jams::dot(layer_normal_unit, globals::lattice->lattice_site_position_cart(spin_index))
      * globals::lattice->parameter() * kMeterToNanometer;
}

std::map<double, LayerBuildData>::iterator find_or_insert_tolerant_layer(
    std::map<double, LayerBuildData>& layers,
    const double position_nm,
    const double distance_tolerance) {
  const auto first_candidate = layers.lower_bound(position_nm - distance_tolerance);
  if (first_candidate != layers.end()
      && std::abs(first_candidate->first - position_nm) <= distance_tolerance) {
    return first_candidate;
  }

  return layers.emplace(position_nm, LayerBuildData{position_nm, {}}).first;
}

std::int64_t checked_layer_bin_index(
    const libconfig::Setting& settings,
    const double position_nm,
    const double z_min,
    const double layer_thickness) {
  const auto bin = std::floor((position_nm - z_min) / layer_thickness);
  if (!std::isfinite(bin)
      || bin < static_cast<double>(std::numeric_limits<std::int64_t>::min())
      || bin > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw jams::ConfigException(settings, "layer bin index exceeds int64 range");
  }
  return static_cast<std::int64_t>(bin);
}

std::vector<LayerBuildData> build_zero_thickness_layers(
    const jams::monitors::SpinGroup& spin_group,
    const jams::Vec<double, 3>& layer_normal_unit,
    const double distance_tolerance) {
  std::map<double, LayerBuildData> layers;

  for (auto spin_index : spin_group.indices_span()) {
    const auto position_nm = projected_layer_position_nm(layer_normal_unit, spin_index);
    auto layer_it = find_or_insert_tolerant_layer(layers, position_nm, distance_tolerance);
    layer_it->second.spin_indices.push_back(spin_index);
  }

  std::vector<LayerBuildData> ordered_layers;
  ordered_layers.reserve(layers.size());
  for (auto& [_, layer] : layers) {
    ordered_layers.push_back(std::move(layer));
  }
  return ordered_layers;
}

std::vector<LayerBuildData> build_finite_thickness_layers(
    const libconfig::Setting& settings,
    const jams::monitors::SpinGroup& spin_group,
    const jams::Vec<double, 3>& layer_normal_unit,
    const double layer_thickness) {
  if (spin_group.empty()) {
    return {};
  }

  double z_min = std::numeric_limits<double>::max();
  for (auto spin_index : spin_group.indices_span()) {
    z_min = std::min(z_min, projected_layer_position_nm(layer_normal_unit, spin_index));
  }

  std::map<std::int64_t, LayerBuildData> layers;
  for (auto spin_index : spin_group.indices_span()) {
    const auto position_nm = projected_layer_position_nm(layer_normal_unit, spin_index);
    const auto bin_index = checked_layer_bin_index(settings, position_nm, z_min, layer_thickness);
    const auto layer_position_nm = z_min + (static_cast<double>(bin_index) + 0.5) * layer_thickness;
    auto [layer_it, _] = layers.emplace(bin_index, LayerBuildData{layer_position_nm, {}});
    layer_it->second.spin_indices.push_back(spin_index);
  }

  std::vector<LayerBuildData> ordered_layers;
  ordered_layers.reserve(layers.size());
  for (auto& [_, layer] : layers) {
    ordered_layers.push_back(std::move(layer));
  }
  return ordered_layers;
}
}

MagnetisationLayersMonitor::MagnetisationLayersMonitor(
    const libconfig::Setting &settings)
    : Monitor(settings) {

  jams::Vec<double, 3> layer_normal = jams::config_required<jams::Vec<double, 3>>(settings, "layer_normal");
  auto layer_thickness = jams::config_optional<double>(settings, "layer_thickness", 0.0);
  auto distance_tolerance = jams::config_optional<double>(settings, "distance_tolerance", jams::defaults::lattice_tolerance);
  validate_layer_normal(settings, layer_normal);
  validate_non_negative_finite_setting(settings, layer_thickness, "layer_thickness");
  validate_non_negative_finite_setting(settings, distance_tolerance, "distance_tolerance");
  const auto layer_normal_unit = jams::unit_vector(layer_normal);

  grouping_ = jams::monitors::parse_spin_grouping(settings, "materials", "magnetisation");
  spin_groups_ = jams::monitors::make_spin_groups(grouping_);
  h5_group_root_name_ = "/jams/monitors/" + name() + "/";

  auto num_groups = spin_groups_.size();
  group_num_layers_.resize(num_groups);
  group_layer_spin_indices_.resize(num_groups);
  group_layer_magnetisation_.resize(num_groups);

  // Create a new h5 file, truncating any old file if it exists.
  HighFive::File file(jams::output::monitor_filename(name(), "h5"),
                      HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {
    const auto& spin_group = spin_groups_[group_idx];

    const auto layers = layer_thickness == 0.0
        ? build_zero_thickness_layers(spin_group, layer_normal_unit, distance_tolerance)
        : build_finite_thickness_layers(settings, spin_group, layer_normal_unit, layer_thickness);

    auto num_layers = layers.size();
    group_num_layers_[group_idx] = num_layers;
    group_layer_spin_indices_[group_idx].resize(num_layers);
    group_layer_magnetisation_[group_idx].resize(num_layers, 3);

    // Move all the data into MultiArrays
    jams::MultiArray<double, 1> layer_positions(num_layers);
    jams::MultiArray<double, 1> layer_saturation_moment(num_layers);
    jams::MultiArray<int, 1> layer_spin_count(num_layers);

    const auto moments = globals::mus.host_view();
    int counter = 0;
    for (auto const &layer: layers) {
      layer_positions(counter) = layer.position_nm;
      layer_spin_count(counter) = layer.spin_indices.size();
      group_layer_spin_indices_[group_idx][counter] = jams::MultiArray<int, 1>(layer.spin_indices);

      layer_saturation_moment(counter) = 0.0;
      for (const auto spin_index : group_layer_spin_indices_[group_idx][counter].host_span()) {
        layer_saturation_moment(counter) += moments(spin_index) / kBohrMagnetonIU;
      }

      ++counter;
    }

    HighFive::Group h5_group = file.createGroup(h5_group_root_name_ +"/groups/" + spin_group.name + "/");
    {
      auto dataset = h5_group.createDataSet<int>(
          "num_layers",HighFive::DataSpace::From(group_num_layers_[group_idx]));
      dataset.write(group_num_layers_[group_idx]);
    }
    {
      auto dataset = h5_group.createDataSet<double>(
          "layer_normal",HighFive::DataSpace::From(layer_normal.values));
      dataset.write(layer_normal.values);
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
      jams::output::monitor_filename(name(), "h5"), HighFive::File::ReadWrite);

  HighFive::Group timeseries_group = file.createGroup(h5_group_root_name_ + "/timeseries/" +  zero_pad_number(solver.iteration(),9));

  timeseries_group.createAttribute<double>("time", solver.time());
  timeseries_group.createAttribute<double>("time_step", solver.time_step());
  timeseries_group.createAttribute<std::string>("units", "ps");

  const auto& spins = globals::s;
  const auto& moments = globals::mus;

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {

    auto spin_group = timeseries_group.createGroup(spin_groups_[group_idx].name);

    // Loop over layers and calculate the magnetisation
    for (auto layer_index = 0; layer_index < group_num_layers_[group_idx]; ++layer_index) {
      jams::Vec<double, 3> mag = jams::sum_spins_moments(spins, moments,
                                           group_layer_spin_indices_[group_idx][layer_index]);

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
