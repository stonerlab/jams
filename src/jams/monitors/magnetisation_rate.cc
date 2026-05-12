// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include "jams/helpers/consts.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"

#include "magnetisation_rate.h"

MagnetisationRateMonitor::MagnetisationRateMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  grouping_(jams::monitors::parse_spin_grouping(settings, "materials", "magnetisation-rate")),
  spin_groups_(jams::monitors::make_spin_groups(grouping_)),
  precision_(jams::config_optional<int>(settings, "precision", 8)),
  tsv_(make_tsv_writer()),
  magnetisation_stats_(),
  convergence_geweke_diagnostic_(100.0)   // number much larger than 1
{}

void MagnetisationRateMonitor::update(Solver& solver) {
  const auto ds_dt = globals::ds_dt.host_view();
  std::vector<jams::Vec<double, 3>> dm_dt(spin_groups_.size(), {0.0, 0.0, 0.0});

  for (std::size_t group_index = 0; group_index < spin_groups_.size(); ++group_index) {
    const auto& group = spin_groups_[group_index];
    for (const auto spin_index : group.indices_span()) {
      for (auto j = 0; j < 3; ++j) {
        dm_dt[group_index][j] += ds_dt(spin_index, j);
      }
    }

    if (!group.empty()) {
      for (auto j = 0; j < 3; ++j) {
        dm_dt[group_index][j] /= static_cast<double>(group.size());
      }
    }
  }

  std::vector<double> values;
  values.reserve(tsv_.num_cols());
  values.push_back(solver.time());

  for (std::size_t type = 0; type < spin_groups_.size(); ++type) {
    for (auto j = 0; j < 3; ++j) {
      values.push_back(dm_dt[type][j]);
    }
  }

    if (convergence_status_ != Monitor::ConvergenceStatus::kDisabled) {
      double total_dm_dt = 0.0;
      for (std::size_t type = 0; type < spin_groups_.size(); ++type) {
        total_dm_dt += jams::norm(dm_dt[type]);
      }

      magnetisation_stats_.add(total_dm_dt);
      double nse = 0.0;
      magnetisation_stats_.geweke(convergence_geweke_diagnostic_, nse);
      values.push_back(convergence_geweke_diagnostic_);
    }

  tsv_.write_row(values);
}

Monitor::ConvergenceStatus MagnetisationRateMonitor::convergence_status() {
  if (convergence_status_ == ConvergenceStatus::kDisabled) {
    return convergence_status_;
  }

  convergence_status_ = std::abs(convergence_geweke_diagnostic_) < convergence_tolerance_
      ? ConvergenceStatus::kConverged
      : ConvergenceStatus::kNotConverged;
  return convergence_status_;
}

jams::output::TsvWriter MagnetisationRateMonitor::make_tsv_writer() const {
  std::vector<jams::output::ColDef> cols;
  cols.push_back({"time", "picoseconds"});

  for (const auto& group : spin_groups_) {
    for (const auto& component : {"dmx_dt", "dmy_dt", "dmz_dt"}) {
      cols.push_back({
          jams::monitors::grouped_column_name(grouping_, group.name, component),
          "ps^-1"});
    }
  }

  if (convergence_status_ != ConvergenceStatus::kDisabled) {
    cols.push_back({"geweke_abs_dm_dt", "dimensionless"});
  }

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols),
      precision_);
}
