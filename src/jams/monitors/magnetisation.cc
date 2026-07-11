// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <sstream>
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
#include "jams/monitors/cuda_grouped_spin_reduction.h"
#endif

#if HAS_CUDA
struct MagnetisationMonitor::CudaBackend {
  jams::monitors::CudaSpinGroupChunks chunks;
  CudaStream stream;
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
  cuda_backend_->chunks = jams::monitors::make_cuda_spin_group_chunks(spin_groups_);
}

void MagnetisationMonitor::accumulate_magnetisation_cuda() {
  prepare_cuda_backend();

  const auto& chunks = cuda_backend_->chunks;
  jams::monitors::execute_cuda_grouped_spin_moment_reduction(
      cuda_backend_->stream,
      chunks.num_groups,
      chunks.num_chunks,
      chunks.chunk_begin_offsets.device_data(),
      chunks.chunk_end_offsets.device_data(),
      chunks.chunk_group_indices.device_data(),
      chunks.spin_indices.device_data(),
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
      const double group_moment =
          jams::scalar_field_indexed_reduce(globals::mus, group.indices_array());
      if (group_moment == 0.0) {
        group_normalising_factors_.push_back(0.0);
        continue;
      }
      group_normalising_factors_.push_back(1.0 / group_moment);
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
