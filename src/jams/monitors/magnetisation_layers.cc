// magnetisation_layers.cc                                             -*-C++-*-
#include <jams/monitors/magnetisation_layers.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/maths.h>
#include <jams/helpers/exception.h>
#include <jams/helpers/output.h>
#include <jams/interface/highfive.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace {
struct LayerBuildData {
  double position_nm = 0.0;
  std::vector<int> local_spin_offsets;
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

int checked_int_count(
    const libconfig::Setting& settings,
    const std::size_t count,
    const char* quantity) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw jams::ConfigException(settings, quantity, " exceeds int range");
  }
  return static_cast<int>(count);
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

double default_distance_tolerance_nm() {
  // The shared lattice tolerance is expressed in lattice-parameter units, but
  // layer coordinates are projected and stored in nm. Convert the default at the
  // monitor boundary so explicit distance_tolerance settings can remain in nm.
  return jams::defaults::lattice_tolerance
      * globals::lattice->parameter()
      * kMeterToNanometer;
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
    const double layer_thickness,
    const double distance_tolerance) {
  const auto scaled_position = (position_nm - z_min) / layer_thickness;
  if (!std::isfinite(scaled_position)) {
    throw jams::ConfigException(settings, "layer bin coordinate exceeds finite range");
  }

  auto bin = std::floor(scaled_position);
  const auto nearest_boundary = std::round(scaled_position);
  const auto boundary_distance_nm = std::abs(scaled_position - nearest_boundary) * layer_thickness;
  if (boundary_distance_nm <= distance_tolerance) {
    // Exact layer boundaries conventionally belong to the upper bin because
    // floor(n) == n. Snap near-boundary round-off to the same convention.
    bin = nearest_boundary;
  }

  if (bin < static_cast<double>(std::numeric_limits<std::int64_t>::min())
      || bin > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw jams::ConfigException(settings, "layer bin index exceeds int64 range");
  }
  return static_cast<std::int64_t>(bin);
}

std::vector<LayerBuildData> build_zero_thickness_layers(
    const libconfig::Setting& settings,
    const jams::monitors::SpinGroup& spin_group,
    const jams::Vec<double, 3>& layer_normal_unit,
    const double distance_tolerance) {
  if (spin_group.empty()) {
    return {};
  }

  double z_min = std::numeric_limits<double>::max();
  for (auto spin_index : spin_group.indices_span()) {
    z_min = std::min(z_min, projected_layer_position_nm(layer_normal_unit, spin_index));
  }

  std::map<double, LayerBuildData> layers;

  const auto spin_indices = spin_group.indices_span();
  for (std::size_t local_offset = 0; local_offset < spin_indices.size(); ++local_offset) {
    const auto spin_index = spin_indices[local_offset];
    const auto position_nm = projected_layer_position_nm(layer_normal_unit, spin_index);
    // Store strict map keys relative to the group minimum. This keeps tolerant
    // comparisons away from large absolute coordinates while preserving the
    // absolute layer position that is written to HDF5.
    const auto relative_position_nm = position_nm - z_min;
    auto layer_it = find_or_insert_tolerant_layer(layers, relative_position_nm, distance_tolerance);
    if (layer_it->second.local_spin_offsets.empty()) {
      layer_it->second.position_nm = z_min + layer_it->first;
    }
    layer_it->second.local_spin_offsets.push_back(
        checked_int_count(settings, local_offset, "spin group local offset"));
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
    const double layer_thickness,
    const double distance_tolerance) {
  if (spin_group.empty()) {
    return {};
  }

  double z_min = std::numeric_limits<double>::max();
  for (auto spin_index : spin_group.indices_span()) {
    z_min = std::min(z_min, projected_layer_position_nm(layer_normal_unit, spin_index));
  }

  std::map<std::int64_t, LayerBuildData> layers;
  const auto spin_indices = spin_group.indices_span();
  for (std::size_t local_offset = 0; local_offset < spin_indices.size(); ++local_offset) {
    const auto spin_index = spin_indices[local_offset];
    const auto position_nm = projected_layer_position_nm(layer_normal_unit, spin_index);
    const auto bin_index = checked_layer_bin_index(
        settings,
        position_nm,
        z_min,
        layer_thickness,
        distance_tolerance);
    const auto layer_position_nm = z_min + (static_cast<double>(bin_index) + 0.5) * layer_thickness;
    auto [layer_it, _] = layers.emplace(bin_index, LayerBuildData{layer_position_nm, {}});
    layer_it->second.local_spin_offsets.push_back(
        checked_int_count(settings, local_offset, "spin group local offset"));
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
  auto distance_tolerance = jams::config_optional<double>(
      settings,
      "distance_tolerance",
      default_distance_tolerance_nm());
  validate_layer_normal(settings, layer_normal);
  validate_non_negative_finite_setting(settings, layer_thickness, "layer_thickness");
  validate_non_negative_finite_setting(settings, distance_tolerance, "distance_tolerance");
  const auto layer_normal_unit = jams::unit_vector(layer_normal);

  grouping_ = jams::monitors::parse_spin_grouping(settings, "materials", "magnetisation");
  spin_groups_ = jams::monitors::make_spin_groups(grouping_);
  h5_group_root_name_ = "/jams/monitors/" + name() + "/";

  auto num_groups = spin_groups_.size();
  group_num_layers_.resize(num_groups);
  group_spin_layer_indices_.resize(num_groups);
  group_layer_magnetisation_.resize(num_groups);

  // Create a new h5 file, truncating any old file if it exists.
  HighFive::File file(jams::output::monitor_filename(name(), "h5"),
                      HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {
    const auto& spin_group = spin_groups_[group_idx];

    const auto layers = layer_thickness == 0.0
        ? build_zero_thickness_layers(settings, spin_group, layer_normal_unit, distance_tolerance)
        : build_finite_thickness_layers(
            settings,
            spin_group,
            layer_normal_unit,
            layer_thickness,
            distance_tolerance);

    auto num_layers = layers.size();
    group_num_layers_[group_idx] = checked_int_count(settings, num_layers, "number of magnetisation layers");
    group_spin_layer_indices_[group_idx].resize(spin_group.size());
    group_layer_magnetisation_[group_idx].resize(num_layers, 3);

    // Move all the data into MultiArrays
    jams::MultiArray<double, 1> layer_positions(num_layers);
    jams::MultiArray<double, 1> layer_saturation_moment(num_layers);
    jams::MultiArray<int, 1> layer_spin_count(num_layers);

    const auto moments = globals::mus.host_view();
    const auto spin_indices = spin_group.indices_span();
    auto spin_layer_indices = group_spin_layer_indices_[group_idx].mutable_host_span();
    int counter = 0;
    for (auto const &layer: layers) {
      layer_positions(counter) = layer.position_nm;
      layer_spin_count(counter) = checked_int_count(settings, layer.local_spin_offsets.size(), "layer spin count");

      layer_saturation_moment(counter) = 0.0;
      for (const auto local_offset : layer.local_spin_offsets) {
        spin_layer_indices[local_offset] = counter;
        const auto spin_index = spin_indices[local_offset];
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
  const auto spin_values = spins.host_view();
  const auto moment_values = moments.host_view();

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {

    auto spin_group = timeseries_group.createGroup(spin_groups_[group_idx].name);

    group_layer_magnetisation_[group_idx].zero();
    auto& layer_magnetisation = group_layer_magnetisation_[group_idx];
    const auto spin_indices = spin_groups_[group_idx].indices_span();
    const auto spin_layer_indices = group_spin_layer_indices_[group_idx].host_span();

    // Accumulate all layer magnetisations in one pass over the group. This
    // avoids storing one spin-index array per layer and avoids rescanning the
    // group once for every layer on each monitor update.
    for (std::size_t n = 0; n < spin_indices.size(); ++n) {
      const auto spin_index = spin_indices[n];
      const auto layer_index = spin_layer_indices[n];
      const auto moment_mu_b = moment_values(spin_index) / kBohrMagnetonIU;

      layer_magnetisation(layer_index, 0) += moment_mu_b * spin_values(spin_index, 0);
      layer_magnetisation(layer_index, 1) += moment_mu_b * spin_values(spin_index, 1);
      layer_magnetisation(layer_index, 2) += moment_mu_b * spin_values(spin_index, 2);
    }

    auto dataset = spin_group.createDataSet<double>(
        "magnetisation",HighFive::DataSpace::From(group_layer_magnetisation_[group_idx]));
    dataset.createAttribute<std::string>("axis0", "layer_index");
    dataset.createAttribute<std::string>("axis1", "magnetisation_xyz");
    dataset.createAttribute<std::string>("units", "bohr_magneton");

    dataset.write(group_layer_magnetisation_[group_idx]);
  }
}
