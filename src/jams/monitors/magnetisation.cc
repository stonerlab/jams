// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/output.h"

#include "jams/monitors/magnetisation.h"
#include "jams/helpers/spinops.h"
#include "jams/helpers/array_ops.h"
#include "jams/helpers/container_utils.h"

#if HAS_CUDA
#include "jams/cuda/cuda_stream.h"
#include "jams/monitors/cuda_magnetisation_kernel.h"
#endif

namespace {
#if HAS_CUDA
constexpr std::size_t kCudaMagnetisationChunkSize = 256;

int checked_int_count_runtime(
    const std::size_t count,
    const char* quantity) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(quantity) + " exceeds int range");
  }
  return static_cast<int>(count);
}

void copy_int_vector_to_device_only(
    jams::MultiArray<int, 1>& target,
    const std::vector<int>& values) {
  target.resize(values.size());
  if (values.empty()) {
    return;
  }

  auto target_values = target.mutable_host_span();
  std::copy(values.begin(), values.end(), target_values.begin());
  target.device_data();
  target.release_stale_host();
}
#endif
}  // namespace

#if HAS_CUDA
struct MagnetisationMonitor::CudaBackend {
  jams::MultiArray<int, 1> spin_indices;
  jams::MultiArray<int, 1> chunk_begin_offsets;
  jams::MultiArray<int, 1> chunk_end_offsets;
  jams::MultiArray<int, 1> chunk_group_indices;
  CudaStream stream;

  void build_from_spin_groups(const std::vector<jams::monitors::SpinGroup>& spin_groups) {
    std::vector<int> all_spin_indices;
    std::vector<int> all_chunk_begin_offsets;
    std::vector<int> all_chunk_end_offsets;
    std::vector<int> all_chunk_group_indices;

    std::size_t total_spins = 0;
    for (const auto& group : spin_groups) {
      total_spins += group.size();
    }
    all_spin_indices.reserve(total_spins);

    for (std::size_t group_index = 0; group_index < spin_groups.size(); ++group_index) {
      const auto group_begin = all_spin_indices.size();
      const auto group_spin_indices = spin_groups[group_index].indices_span();
      all_spin_indices.insert(
          all_spin_indices.end(),
          group_spin_indices.begin(),
          group_spin_indices.end());
      const auto group_end = all_spin_indices.size();

      for (auto chunk_begin = group_begin;
           chunk_begin < group_end;
           chunk_begin += kCudaMagnetisationChunkSize) {
        const auto chunk_end = std::min(chunk_begin + kCudaMagnetisationChunkSize, group_end);
        all_chunk_begin_offsets.push_back(
            checked_int_count_runtime(chunk_begin, "cuda magnetisation chunk begin"));
        all_chunk_end_offsets.push_back(
            checked_int_count_runtime(chunk_end, "cuda magnetisation chunk end"));
        all_chunk_group_indices.push_back(
            checked_int_count_runtime(group_index, "cuda magnetisation chunk group index"));
      }
    }

    copy_int_vector_to_device_only(spin_indices, all_spin_indices);
    copy_int_vector_to_device_only(chunk_begin_offsets, all_chunk_begin_offsets);
    copy_int_vector_to_device_only(chunk_end_offsets, all_chunk_end_offsets);
    copy_int_vector_to_device_only(chunk_group_indices, all_chunk_group_indices);
  }
};
#endif

MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_(make_tsv_writer(settings))
{}

MagnetisationMonitor::~MagnetisationMonitor() = default;

void MagnetisationMonitor::update(Solver& solver) {
  const auto& spins = globals::s;
  const auto& moments = globals::mus;
  auto values = make_reserved<double>(tsv_.num_cols());

  solver.append_monitor_coordinates(values);

#if HAS_CUDA
  if (solver.is_cuda_solver()) {
    accumulate_magnetisation_cuda();
    const auto magnetisation = group_magnetisation_.host_view();
    for (std::size_t group_index = 0; group_index < spin_groups_.size(); ++group_index) {
      append_magnetisation_values(
          values,
          {magnetisation(group_index, 0), magnetisation(group_index, 1), magnetisation(group_index, 2)},
          group_index);
    }
    group_magnetisation_.release_stale_host();
    tsv_.write_row(values);
    return;
  }
#endif

  for (std::size_t group_index = 0; group_index < spin_groups_.size(); ++group_index) {
    const auto& group = spin_groups_[group_index];
    if (group.empty()) {
      values.push_back(0.0);
      values.push_back(0.0);
      values.push_back(0.0);
      values.push_back(0.0);
      continue;
    }

    jams::Vec<double, 3> mag = jams::sum_spins_moments(spins, moments, group.indices_array());
    append_magnetisation_values(values, mag, group_index);
  }

  tsv_.write_row(values);
}

void MagnetisationMonitor::append_magnetisation_values(
    std::vector<double>& values,
    const jams::Vec<double, 3>& magnetisation,
    const std::size_t group_index) const {
  const auto normalising_factor = group_normalising_factors_[group_index];
  values.push_back(magnetisation[0] * normalising_factor);
  values.push_back(magnetisation[1] * normalising_factor);
  values.push_back(magnetisation[2] * normalising_factor);
  values.push_back(jams::norm(magnetisation) * normalising_factor);
}

#if HAS_CUDA
void MagnetisationMonitor::prepare_cuda_backend() {
  if (cuda_backend_ != nullptr) {
    return;
  }

  cuda_backend_ = std::make_unique<CudaBackend>();
  cuda_backend_->build_from_spin_groups(spin_groups_);
}

void MagnetisationMonitor::accumulate_magnetisation_cuda() {
  prepare_cuda_backend();

  execute_cuda_magnetisation_kernel(
      cuda_backend_->stream,
      checked_int_count_runtime(spin_groups_.size(), "number of magnetisation groups"),
      checked_int_count_runtime(
          cuda_backend_->chunk_group_indices.size(),
          "number of cuda magnetisation chunks"),
      cuda_backend_->chunk_begin_offsets.device_data(),
      cuda_backend_->chunk_end_offsets.device_data(),
      cuda_backend_->chunk_group_indices.device_data(),
      cuda_backend_->spin_indices.device_data(),
      globals::s.device_data(),
      globals::mus.device_data(),
      group_magnetisation_.mutable_device_data());

  cuda_backend_->stream.synchronize();
}
#endif


jams::output::TsvWriter MagnetisationMonitor::make_tsv_writer(const libconfig::Setting &settings) {
  grouping_ = jams::monitors::parse_spin_grouping(settings, "materials", "magnetisation");
  spin_groups_ = jams::monitors::make_spin_groups(grouping_);
  group_magnetisation_.resize(spin_groups_.size(), 3);

  // should the magnetisation be normalised to 1 or be in units of muB
  normalize_magnetisation_ = jams::config_optional<bool>(settings, "normalize", true);
  group_normalising_factors_.clear();
  group_normalising_factors_.reserve(spin_groups_.size());
  for (const auto& group : spin_groups_) {
    if (group.empty()) {
      group_normalising_factors_.push_back(1.0);
    } else if (normalize_magnetisation_) {
      group_normalising_factors_.push_back(
          1.0 / jams::scalar_field_indexed_reduce(globals::mus, group.indices_array()));
    } else {
      // internally we use meV T^-1 for mus so convert back to Bohr magneton
      group_normalising_factors_.push_back(1.0 / kBohrMagnetonIU);
    }
  }

  auto precision = jams::config_optional<int>(settings, "precision", 8);
  auto cols = globals::solver->monitor_coordinate_columns();

  std::string mag_unit = "dimensionless";
  if (!normalize_magnetisation_) {
    mag_unit = "bohr magnetons";
  }

  for (const auto& group : spin_groups_) {
    for (const auto& component : {"mx", "my", "mz", "m"}) {
      cols.push_back({
          jams::monitors::grouped_column_name(grouping_, group.name, component),
          mag_unit});
    }
  }

  return jams::output::TsvWriter(
    jams::output::monitor_filename(name(), "tsv"),
    std::move(cols),
    precision
  );
}
