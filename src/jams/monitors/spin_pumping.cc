// Copyright 2014 Joseph Barker. All rights reserved.


#include "spin_pumping.h"

#include <string>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/helpers/stats.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"

SpinPumpingMonitor::SpinPumpingMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  grouping_(jams::monitors::parse_spin_grouping(settings, "materials", "spin-pumping")),
  spin_groups_(jams::monitors::make_spin_groups(grouping_)),
  precision_(jams::config_optional<int>(settings, "precision", 8)),
  tsv_(make_tsv_writer())
{
  s_old_.resize(globals::num_spins,3);
}

void SpinPumpingMonitor::update(Solver& solver) {
  const auto spins = globals::s.host_view();

  std::vector<jams::Vec<double, 3>> spin_pumping_real(spin_groups_.size());
  std::vector<jams::Vec<double, 3>> spin_pumping_imag(spin_groups_.size());
  double d_timestep = 1.0/solver.time_step();

  for (std::size_t group_index = 0; group_index < spin_groups_.size(); ++group_index) {
    for (const auto i : spin_groups_[group_index].indices_span()) {
      jams::Vec<double, 3> s_i = {spins(i,0), spins(i, 1), spins(i,2)};
      jams::Vec<double, 3> s_old_i = {s_old_(i,0), s_old_(i, 1), s_old_(i,2)};
      jams::Vec<double, 3> ds_dt_i = (s_i - s_old_i) * d_timestep;

      spin_pumping_real[group_index] += jams::cross(s_i, ds_dt_i);
      spin_pumping_imag[group_index] += ds_dt_i;
    }
  }

  std::vector<double> values;
  values.reserve(tsv_.num_cols());
  solver.append_monitor_coordinates(values);

  for (std::size_t type = 0; type < spin_groups_.size(); ++type) {
    auto norm = spin_groups_[type].empty()
        ? 0.0
        : 1.0 / static_cast<double>(spin_groups_[type].size());
    for (auto j = 0; j < 3; ++j) {
      values.push_back(spin_pumping_real[type][j] * norm);
    }
    for (auto j = 0; j < 3; ++j) {
      values.push_back(spin_pumping_imag[type][j] * norm);
    }
  }

  tsv_.write_row(values);
}

jams::output::TsvWriter SpinPumpingMonitor::make_tsv_writer() const {
  auto cols = globals::solver->monitor_coordinate_columns();

  for (const auto& group : spin_groups_) {
    for (const auto& component : {"Re_J_x", "Re_J_y", "Re_J_z", "Im_J_x", "Im_J_y", "Im_J_z"}) {
      cols.push_back({
          jams::monitors::grouped_column_name(grouping_, group.name, component),
          "rad s^-1 T^-1"});
    }
  }

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols),
      precision_);
}


bool SpinPumpingMonitor::is_updating(const int &iteration) {
  if ((iteration + 1) % output_step_freq_ == 0) {
    const auto& spins = globals::s;
    s_old_ = spins;
  }
  if (iteration % output_step_freq_ == 0) {
    return true;
  }
  return false;
}
