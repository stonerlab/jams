// magnetisation_layers.cc                                             -*-C++-*-
#include <jams/monitors/magnetisation_layers.h>

#include <jams/core/globals.h>
#include <jams/core/lattice.h>
#include <jams/core/solver.h>
#include <jams/helpers/maths.h>
#include <jams/helpers/exception.h>
#include <jams/helpers/output.h>
#include <jams/helpers/utils.h>
#include <jams/interface/highfive.h>

#if HAS_CUDA
#include <jams/cuda/cuda_stream.h>
#include <jams/monitors/cuda_magnetisation_layers_kernel.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct LayerBuildData {
  double position_nm = 0.0;
  std::vector<int> local_spin_offsets;
};

struct PlaneBasis {
  jams::Vec<double, 3> u{1.0, 0.0, 0.0};
  jams::Vec<double, 3> v{0.0, 1.0, 0.0};
};

struct LayerVolumeBounds {
  double lower_nm = 0.0;
  double upper_nm = 0.0;
};

#if HAS_CUDA
constexpr std::size_t kCudaLayerChunkSize = 256;

void copy_int_vector_to_device_only(
    jams::MultiArray<int, 1>& target,
    const std::vector<int>& values) {
  target.resize(values.size());
  if (values.empty()) {
    return;
  }

  auto target_values = target.mutable_host_span();
  std::copy(values.begin(), values.end(), target_values.begin());
  // Static CUDA work arrays are never read on the host after construction.
  // Copy them once, then release the duplicate host allocation.
  target.device_data();
  target.release_stale_host();
}
#endif

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

int checked_int_count_runtime(
    const std::size_t count,
    const char* quantity) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(quantity) + " exceeds int range");
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

std::string xml_escape(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char ch : text) {
    switch (ch) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&apos;";
        break;
      default:
        escaped += ch;
        break;
    }
  }
  return escaped;
}

PlaneBasis plane_basis_from_normal(const jams::Vec<double, 3>& normal) {
  const auto reference = std::abs(normal[0]) < 0.9
      ? jams::Vec<double, 3>{1.0, 0.0, 0.0}
      : jams::Vec<double, 3>{0.0, 1.0, 0.0};
  const auto u = jams::unit_vector(jams::cross(normal, reference));
  const auto v = jams::unit_vector(jams::cross(normal, u));
  return {u, v};
}

std::array<jams::Vec<double, 3>, 8> supercell_corners_nm() {
  const auto& supercell = globals::lattice->get_supercell();
  const double scale = globals::lattice->parameter() * kMeterToNanometer;
  const auto a1 = supercell.a1() * scale;
  const auto a2 = supercell.a2() * scale;
  const auto a3 = supercell.a3() * scale;

  return {
      jams::Vec<double, 3>{0.0, 0.0, 0.0},
      a1,
      a2,
      a3,
      a1 + a2,
      a1 + a3,
      a2 + a3,
      a1 + a2 + a3,
  };
}

double projected_sample_thickness_nm(
    const std::array<jams::Vec<double, 3>, 8>& corners,
    const jams::Vec<double, 3>& layer_normal_unit) {
  double lower = std::numeric_limits<double>::max();
  double upper = std::numeric_limits<double>::lowest();
  for (const auto& corner : corners) {
    const auto projection = jams::dot(corner, layer_normal_unit);
    lower = std::min(lower, projection);
    upper = std::max(upper, projection);
  }
  return upper - lower;
}

std::vector<LayerVolumeBounds> build_layer_volume_bounds(
    const jams::MultiArray<double, 1>& layer_positions,
    const double layer_thickness,
    const double single_layer_fallback_thickness) {
  const auto num_layers = layer_positions.size();
  std::vector<LayerVolumeBounds> bounds(num_layers);
  const auto layer_position_values = layer_positions.host_view();

  if (layer_thickness > 0.0) {
    const auto half_thickness = 0.5 * layer_thickness;
    for (std::size_t layer = 0; layer < num_layers; ++layer) {
      bounds[layer] = {
          layer_position_values(layer) - half_thickness,
          layer_position_values(layer) + half_thickness,
      };
    }
    return bounds;
  }

  if (num_layers == 1) {
    const auto half_thickness = 0.5 * single_layer_fallback_thickness;
    bounds[0] = {
        layer_position_values(0) - half_thickness,
        layer_position_values(0) + half_thickness,
    };
    return bounds;
  }

  for (std::size_t layer = 0; layer < num_layers; ++layer) {
    const auto lower = layer == 0
        ? layer_position_values(0) - 0.5 * (layer_position_values(1) - layer_position_values(0))
        : 0.5 * (layer_position_values(layer - 1) + layer_position_values(layer));
    const auto upper = layer + 1 == num_layers
        ? layer_position_values(layer) + 0.5 * (layer_position_values(layer) - layer_position_values(layer - 1))
        : 0.5 * (layer_position_values(layer) + layer_position_values(layer + 1));
    bounds[layer] = {lower, upper};
  }

  return bounds;
}

void write_xdmf_volume_and_glyph_geometry(
    HighFive::Group& h5_group,
    const jams::MultiArray<double, 1>& layer_positions,
    const jams::Vec<double, 3>& layer_normal_unit,
    const double layer_thickness) {
  const auto num_layers = layer_positions.size();
  jams::MultiArray<double, 2> volume_points(num_layers * 8, 3);
  jams::MultiArray<int, 2> volume_cells(num_layers, 8);
  jams::MultiArray<double, 2> glyph_points(num_layers, 3);

  const auto basis = plane_basis_from_normal(layer_normal_unit);
  const auto corners = supercell_corners_nm();
  const auto sample_thickness = projected_sample_thickness_nm(corners, layer_normal_unit);
  const auto single_layer_fallback_thickness = definately_greater_than(
      sample_thickness,
      0.0,
      std::numeric_limits<double>::epsilon())
      ? sample_thickness
      : default_distance_tolerance_nm();
  const auto layer_bounds = build_layer_volume_bounds(
      layer_positions,
      layer_thickness,
      single_layer_fallback_thickness);

  double u_min = std::numeric_limits<double>::max();
  double u_max = std::numeric_limits<double>::lowest();
  double v_min = std::numeric_limits<double>::max();
  double v_max = std::numeric_limits<double>::lowest();
  for (const auto& corner : corners) {
    const auto u_projection = jams::dot(corner, basis.u);
    const auto v_projection = jams::dot(corner, basis.v);
    u_min = std::min(u_min, u_projection);
    u_max = std::max(u_max, u_projection);
    v_min = std::min(v_min, v_projection);
    v_max = std::max(v_max, v_projection);
  }

  auto volume_point_values = volume_points.mutable_host_view();
  auto volume_cell_values = volume_cells.mutable_host_view();
  auto glyph_point_values = glyph_points.mutable_host_view();
  const auto layer_position_values = layer_positions.host_view();
  for (std::size_t layer = 0; layer < num_layers; ++layer) {
    const auto lower_center = layer_normal_unit * layer_bounds[layer].lower_nm;
    const auto upper_center = layer_normal_unit * layer_bounds[layer].upper_nm;
    const auto glyph_center = layer_normal_unit * layer_position_values(layer);
    const std::array<jams::Vec<double, 3>, 8> layer_points{
        lower_center + basis.u * u_min + basis.v * v_min,
        lower_center + basis.u * u_max + basis.v * v_min,
        lower_center + basis.u * u_max + basis.v * v_max,
        lower_center + basis.u * u_min + basis.v * v_max,
        upper_center + basis.u * u_min + basis.v * v_min,
        upper_center + basis.u * u_max + basis.v * v_min,
        upper_center + basis.u * u_max + basis.v * v_max,
        upper_center + basis.u * u_min + basis.v * v_max,
    };

    for (auto component = 0; component < 3; ++component) {
      glyph_point_values(layer, component) = glyph_center[component];
    }

    for (auto corner = 0; corner < 8; ++corner) {
      const auto point_index = layer * 8 + corner;
      for (auto component = 0; component < 3; ++component) {
        volume_point_values(point_index, component) = layer_points[corner][component];
      }
      volume_cell_values(layer, corner) = static_cast<int>(point_index);
    }
  }

  auto xdmf_group = h5_group.createGroup("xdmf");
  auto volume_points_dataset = xdmf_group.createDataSet<double>(
      "volume_points", HighFive::DataSpace::From(volume_points));
  volume_points_dataset.write(volume_points);
  volume_points_dataset.createAttribute<std::string>("units", "nm");
  volume_points_dataset.createAttribute<std::string>("axis0", "point_index");
  volume_points_dataset.createAttribute<std::string>("axis1", "xyz");

  auto volume_cells_dataset = xdmf_group.createDataSet<int>(
      "volume_cells", HighFive::DataSpace::From(volume_cells));
  volume_cells_dataset.write(volume_cells);
  volume_cells_dataset.createAttribute<std::string>("axis0", "layer_index");
  volume_cells_dataset.createAttribute<std::string>("axis1", "hexahedron_corner_index");

  auto glyph_points_dataset = xdmf_group.createDataSet<double>(
      "glyph_points", HighFive::DataSpace::From(glyph_points));
  glyph_points_dataset.write(glyph_points);
  glyph_points_dataset.createAttribute<std::string>("units", "nm");
  glyph_points_dataset.createAttribute<std::string>("axis0", "layer_index");
  glyph_points_dataset.createAttribute<std::string>("axis1", "xyz");
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

#if HAS_CUDA
struct MagnetisationLayersMonitor::CudaBackend {
  struct GroupWork {
    jams::MultiArray<int, 1> spin_indices;
    jams::MultiArray<int, 1> chunk_begin_offsets;
    jams::MultiArray<int, 1> chunk_end_offsets;
    jams::MultiArray<int, 1> chunk_layer_indices;
  };

  explicit CudaBackend(const std::size_t num_groups)
      : group_work(num_groups) {}

  void build_group_work_from_layers(
      const std::size_t group_idx,
      const libconfig::Setting& settings,
      const jams::monitors::SpinGroup& spin_group,
      const std::vector<LayerBuildData>& layers) {
    std::vector<int> sorted_spin_indices;
    std::vector<int> chunk_begin_offsets;
    std::vector<int> chunk_end_offsets;
    std::vector<int> chunk_layer_indices;

    sorted_spin_indices.reserve(spin_group.size());
    const auto group_spin_indices = spin_group.indices_span();

    for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
      // Keep spins for each layer contiguous so each CUDA block can reduce one
      // fixed-size chunk without needing per-spin layer lookups.
      const auto layer_begin = sorted_spin_indices.size();
      for (const auto local_offset : layers[layer_idx].local_spin_offsets) {
        sorted_spin_indices.push_back(group_spin_indices[local_offset]);
      }
      const auto layer_end = sorted_spin_indices.size();

      for (auto chunk_begin = layer_begin; chunk_begin < layer_end; chunk_begin += kCudaLayerChunkSize) {
        const auto chunk_end = std::min(chunk_begin + kCudaLayerChunkSize, layer_end);
        chunk_begin_offsets.push_back(checked_int_count(settings, chunk_begin, "cuda layer chunk begin"));
        chunk_end_offsets.push_back(checked_int_count(settings, chunk_end, "cuda layer chunk end"));
        chunk_layer_indices.push_back(checked_int_count(settings, layer_idx, "cuda layer chunk layer index"));
      }
    }

    copy_to_device_only(group_idx, sorted_spin_indices, chunk_begin_offsets, chunk_end_offsets, chunk_layer_indices);
  }

  void build_group_work_from_cpu_indices(
      const std::size_t group_idx,
      const jams::monitors::SpinGroup& spin_group,
      const jams::MultiArray<int, 1>& spin_layer_indices,
      const int num_layers) {
    if (num_layers < 0) {
      throw std::runtime_error("number of cuda magnetisation layers is negative");
    }

    const auto group_spin_indices = spin_group.indices_span();
    const auto layer_indices = spin_layer_indices.host_span();
    const auto num_layers_size = static_cast<std::size_t>(num_layers);

    std::vector<std::size_t> layer_counts(num_layers_size, 0);
    for (const auto layer_index : layer_indices) {
      if (layer_index < 0 || layer_index >= num_layers) {
        throw std::runtime_error("cuda layer index is outside layer range");
      }
      ++layer_counts[static_cast<std::size_t>(layer_index)];
    }

    std::vector<std::size_t> layer_offsets(num_layers_size + 1, 0);
    for (std::size_t layer_idx = 0; layer_idx < num_layers_size; ++layer_idx) {
      layer_offsets[layer_idx + 1] = layer_offsets[layer_idx] + layer_counts[layer_idx];
    }

    std::vector<int> sorted_spin_indices(group_spin_indices.size());
    auto layer_cursors = layer_offsets;
    // This is the lazy CUDA path used if a monitor was constructed before the
    // CUDA solver pointer was available. Rebuild the same layer-contiguous work
    // order from the CPU membership array.
    for (std::size_t n = 0; n < group_spin_indices.size(); ++n) {
      const auto layer_index = static_cast<std::size_t>(layer_indices[n]);
      sorted_spin_indices[layer_cursors[layer_index]++] = group_spin_indices[n];
    }

    std::vector<int> chunk_begin_offsets;
    std::vector<int> chunk_end_offsets;
    std::vector<int> chunk_layer_indices;
    for (std::size_t layer_idx = 0; layer_idx < num_layers_size; ++layer_idx) {
      for (auto chunk_begin = layer_offsets[layer_idx];
           chunk_begin < layer_offsets[layer_idx + 1];
           chunk_begin += kCudaLayerChunkSize) {
        const auto chunk_end = std::min(chunk_begin + kCudaLayerChunkSize, layer_offsets[layer_idx + 1]);
        chunk_begin_offsets.push_back(checked_int_count_runtime(chunk_begin, "cuda layer chunk begin"));
        chunk_end_offsets.push_back(checked_int_count_runtime(chunk_end, "cuda layer chunk end"));
        chunk_layer_indices.push_back(checked_int_count_runtime(layer_idx, "cuda layer chunk layer index"));
      }
    }

    copy_to_device_only(group_idx, sorted_spin_indices, chunk_begin_offsets, chunk_end_offsets, chunk_layer_indices);
  }

  void copy_to_device_only(
      const std::size_t group_idx,
      const std::vector<int>& sorted_spin_indices,
      const std::vector<int>& chunk_begin_offsets,
      const std::vector<int>& chunk_end_offsets,
      const std::vector<int>& chunk_layer_indices) {
    auto& work = group_work[group_idx];
    copy_int_vector_to_device_only(work.spin_indices, sorted_spin_indices);
    copy_int_vector_to_device_only(work.chunk_begin_offsets, chunk_begin_offsets);
    copy_int_vector_to_device_only(work.chunk_end_offsets, chunk_end_offsets);
    copy_int_vector_to_device_only(work.chunk_layer_indices, chunk_layer_indices);
  }

  std::vector<GroupWork> group_work;
  CudaStream stream;
};
#endif

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
  h5_file_name_ = jams::output::monitor_filename(name(), "h5");
  xdmf_file_name_ = jams::output::monitor_filename(name(), "xdmf");

  auto num_groups = spin_groups_.size();
  group_num_layers_.resize(num_groups);
  group_spin_layer_indices_.resize(num_groups);
  group_layer_magnetisation_.resize(num_groups);

#if HAS_CUDA
  if (globals::solver != nullptr && globals::solver->is_cuda_solver()) {
    cuda_backend_ = std::make_unique<CudaBackend>(num_groups);
  }
#endif

  // Create a new h5 file, truncating any old file if it exists.
  HighFive::File file(h5_file_name_,
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
    group_layer_magnetisation_[group_idx].resize(num_layers, 3);

    std::span<int> spin_layer_indices;
#if HAS_CUDA
    const bool use_cuda_backend = cuda_backend_ != nullptr;
    if (!use_cuda_backend) {
#endif
      group_spin_layer_indices_[group_idx].resize(spin_group.size());
      spin_layer_indices = group_spin_layer_indices_[group_idx].mutable_host_span();
#if HAS_CUDA
    }
#endif

    // Move all the data into MultiArrays
    jams::MultiArray<double, 1> layer_positions(num_layers);
    jams::MultiArray<double, 1> layer_saturation_moment(num_layers);
    jams::MultiArray<int, 1> layer_spin_count(num_layers);

    const auto moments = globals::mus.host_view();
    const auto spin_indices = spin_group.indices_span();
    int counter = 0;
    for (auto const &layer: layers) {
      layer_positions(counter) = layer.position_nm;
      layer_spin_count(counter) = checked_int_count(settings, layer.local_spin_offsets.size(), "layer spin count");

      layer_saturation_moment(counter) = 0.0;
      for (const auto local_offset : layer.local_spin_offsets) {
#if HAS_CUDA
        if (!use_cuda_backend) {
#endif
          spin_layer_indices[local_offset] = counter;
#if HAS_CUDA
        }
#endif
        const auto spin_index = spin_indices[local_offset];
        layer_saturation_moment(counter) += moments(spin_index) / kBohrMagnetonIU;
      }

      ++counter;
    }

#if HAS_CUDA
    if (use_cuda_backend) {
      cuda_backend_->build_group_work_from_layers(group_idx, settings, spin_group, layers);
    }
#endif

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
    write_xdmf_volume_and_glyph_geometry(
        h5_group,
        layer_positions,
        layer_normal_unit,
        layer_thickness);
  }

  write_xdmf_file();
}


MagnetisationLayersMonitor::~MagnetisationLayersMonitor() = default;

void MagnetisationLayersMonitor::write_xdmf_file() const {
  std::ofstream xdmf(xdmf_file_name_, std::ios::out | std::ios::trunc);
  if (!xdmf) {
    throw std::runtime_error("failed to open magnetisation-layers XDMF file for writing");
  }

  const auto h5_basename = file_basename(h5_file_name_);
  xdmf << "<?xml version=\"1.0\"?>\n";
  xdmf << "<!DOCTYPE Xdmf SYSTEM \"https://gitlab.kitware.com/xdmf/xdmf/raw/master/Xdmf.dtd\"[]>\n";
  xdmf << "<Xdmf Version=\"3.0\">\n";
  xdmf << "  <Domain Name=\"JAMS\">\n";
  xdmf << "    <Information Name=\"Configuration\" Value=\""
       << xml_escape(globals::simulation_name) << "\" />\n";

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {
    const auto& group = spin_groups_[group_idx];
    const auto group_name_xml = xml_escape(group.name);
    const auto num_layers = group_num_layers_[group_idx];
    const auto num_volume_points = num_layers * 8;
    const auto group_path = h5_group_root_name_ + "groups/" + group.name;

    xdmf << "    <Grid Name=\"" << group_name_xml
         << "_volumes\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for (const auto& step : xdmf_time_steps_) {
      const auto iteration = zero_pad_number(step.iteration, 9);
      const auto time_path = h5_group_root_name_ + "timeseries/" + iteration + "/" + group.name;

      xdmf << "      <Grid Name=\"" << group_name_xml << "_volumes_" << iteration
           << "\" GridType=\"Uniform\">\n";
      xdmf << "        <Time Value=\"" << std::setprecision(17) << step.time << "\" />\n";
      xdmf << "        <Topology TopologyType=\"Hexahedron\" Dimensions=\""
           << num_layers << "\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << " 8\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/xdmf/volume_cells\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Topology>\n";
      xdmf << "        <Geometry GeometryType=\"XYZ\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_volume_points
           << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/xdmf/volume_points\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Geometry>\n";
      xdmf << "        <Attribute Name=\"Magnetisation\" AttributeType=\"Vector\" Center=\"Cell\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << time_path << "/magnetisation\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "        <Attribute Name=\"LayerPosition\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/layer_positions\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "        <Attribute Name=\"SaturationMoment\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/layer_saturation_moment\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "        <Attribute Name=\"SpinCount\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << "\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/layer_spin_count\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "      </Grid>\n";
    }
    xdmf << "    </Grid>\n";

    xdmf << "    <Grid Name=\"" << group_name_xml
         << "_glyphs\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    for (const auto& step : xdmf_time_steps_) {
      const auto iteration = zero_pad_number(step.iteration, 9);
      const auto time_path = h5_group_root_name_ + "timeseries/" + iteration + "/" + group.name;

      xdmf << "      <Grid Name=\"" << group_name_xml << "_glyphs_" << iteration
           << "\" GridType=\"Uniform\">\n";
      xdmf << "        <Time Value=\"" << std::setprecision(17) << step.time << "\" />\n";
      xdmf << "        <Topology TopologyType=\"Polyvertex\" Dimensions=\""
           << num_layers << "\" />\n";
      xdmf << "        <Geometry GeometryType=\"XYZ\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/xdmf/glyph_points\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Geometry>\n";
      xdmf << "        <Attribute Name=\"Magnetisation\" AttributeType=\"Vector\" Center=\"Node\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << time_path << "/magnetisation\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "        <Attribute Name=\"LayerPosition\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/layer_positions\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "        <Attribute Name=\"SaturationMoment\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/layer_saturation_moment\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "        <Attribute Name=\"SpinCount\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      xdmf << "          <DataItem Dimensions=\"" << num_layers
           << "\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n";
      xdmf << "            " << h5_basename << ":" << group_path << "/layer_spin_count\n";
      xdmf << "          </DataItem>\n";
      xdmf << "        </Attribute>\n";
      xdmf << "      </Grid>\n";
    }
    xdmf << "    </Grid>\n";
  }

  xdmf << "  </Domain>\n";
  xdmf << "</Xdmf>\n";
}

void MagnetisationLayersMonitor::append_xdmf_time_step(const Solver& solver) {
  xdmf_time_steps_.push_back({solver.iteration(), solver.time()});
  write_xdmf_file();
}

void MagnetisationLayersMonitor::accumulate_layer_magnetisation_cpu() {
  const auto& spins = globals::s;
  const auto& moments = globals::mus;
  const auto spin_values = spins.host_view();
  const auto moment_values = moments.host_view();

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {
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
  }
}

#if HAS_CUDA
void MagnetisationLayersMonitor::prepare_cuda_backend_from_cpu_indices() {
  cuda_backend_ = std::make_unique<CudaBackend>(spin_groups_.size());
  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {
    // This fallback preserves correctness if update() is first called with a
    // CUDA solver after construction used the CPU membership arrays.
    cuda_backend_->build_group_work_from_cpu_indices(
        group_idx,
        spin_groups_[group_idx],
        group_spin_layer_indices_[group_idx],
        group_num_layers_[group_idx]);
  }
}

void MagnetisationLayersMonitor::accumulate_layer_magnetisation_cuda() {
  if (cuda_backend_ == nullptr) {
    prepare_cuda_backend_from_cpu_indices();
  }

  const auto& spins = globals::s;
  const auto& moments = globals::mus;

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {
    auto& layer_magnetisation = group_layer_magnetisation_[group_idx];
    auto& work = cuda_backend_->group_work[group_idx];

    execute_cuda_magnetisation_layers_kernel(
        cuda_backend_->stream,
        group_num_layers_[group_idx],
        checked_int_count_runtime(work.chunk_layer_indices.size(), "number of cuda layer chunks"),
        work.chunk_begin_offsets.device_data(),
        work.chunk_end_offsets.device_data(),
        work.chunk_layer_indices.device_data(),
        work.spin_indices.device_data(),
        spins.device_data(),
        moments.device_data(),
        layer_magnetisation.mutable_device_data());
  }

  cuda_backend_->stream.synchronize();
}
#endif

void MagnetisationLayersMonitor::update(Solver& solver) {
  // Open the h5 file to write new data
  HighFive::File file(
      h5_file_name_, HighFive::File::ReadWrite);

  HighFive::Group timeseries_group = file.createGroup(h5_group_root_name_ + "/timeseries/" +  zero_pad_number(solver.iteration(),9));

  timeseries_group.createAttribute<double>("time", solver.time());
  timeseries_group.createAttribute<double>("time_step", solver.time_step());
  timeseries_group.createAttribute<std::string>("units", "ps");

#if HAS_CUDA
  const bool using_cuda_backend = solver.is_cuda_solver();
  if (using_cuda_backend) {
    accumulate_layer_magnetisation_cuda();
  } else {
    accumulate_layer_magnetisation_cpu();
  }
#else
  const bool using_cuda_backend = false;
  accumulate_layer_magnetisation_cpu();
#endif

  for (std::size_t group_idx = 0; group_idx < spin_groups_.size(); ++group_idx) {
    auto spin_group = timeseries_group.createGroup(spin_groups_[group_idx].name);

    auto dataset = spin_group.createDataSet<double>(
        "magnetisation",HighFive::DataSpace::From(group_layer_magnetisation_[group_idx]));
    dataset.createAttribute<std::string>("axis0", "layer_index");
    dataset.createAttribute<std::string>("axis1", "magnetisation_xyz");
    dataset.createAttribute<std::string>("units", "bohr_magneton");

    dataset.write(group_layer_magnetisation_[group_idx]);
#if HAS_CUDA
    if (using_cuda_backend) {
      group_layer_magnetisation_[group_idx].release_stale_host();
    }
#else
    (void)using_cuda_backend;
#endif
  }

  append_xdmf_time_step(solver);
}
