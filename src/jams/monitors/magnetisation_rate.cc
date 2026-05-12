// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include "jams/helpers/consts.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

#include "magnetisation_rate.h"

MagnetisationRateMonitor::MagnetisationRateMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::output::full_path_filename("dm_dt.tsv")),
  spin_groups_(jams::monitors::make_spin_groups(jams::monitors::SpinGrouping::MATERIALS)),
  magnetisation_stats_(),
  convergence_geweke_diagnostic_(100.0)   // number much larger than 1
{
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

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

  tsv_file.width(12);
  tsv_file << std::scientific << solver.time() << "\t";

  for (std::size_t type = 0; type < spin_groups_.size(); ++type) {
    for (auto j = 0; j < 3; ++j) {
      tsv_file << dm_dt[type][j] << "\t";
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
      tsv_file << convergence_geweke_diagnostic_;
    }

    tsv_file << std::endl;
}

Monitor::ConvergenceStatus MagnetisationRateMonitor::convergence_status() {
  if (convergence_status_ == ConvergenceStatus::kDisabled) {
    return convergence_status_;
  }

  if (std::abs(convergence_geweke_diagnostic_) < convergence_tolerance_) {
    convergence_status_ = ConvergenceStatus::kConverged;
  }

  return ConvergenceStatus::kNotConverged;
}

std::string MagnetisationRateMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";

  for (const auto& group : spin_groups_) {
    ss << group.name + "_dmx_dt\t";
    ss << group.name + "_dmy_dt\t";
    ss << group.name + "_dmz_dt\t";
  }

  if (convergence_status_ != ConvergenceStatus::kDisabled) {
    ss << "geweke_abs_dm_dt\t";
  }
  ss << std::endl;

  return ss.str();
}
