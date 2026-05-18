// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"

#include "smr.h"

SMRMonitor::SMRMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  grouping_(jams::monitors::parse_spin_grouping(settings, "materials", "smr")),
  spin_groups_(jams::monitors::make_spin_groups(grouping_)),
  precision_(jams::config_optional<int>(settings, "precision", 8)),
  tsv_(make_tsv_writer())
{
  std::cout << "\ninitialising SMR monitor\n";
  std::cout << "  assumes axes j->x, t->y, n->z\n";
}

void SMRMonitor::update(Solver& solver) {
  const auto spins = globals::s.host_view();

  std::vector<double> mtsq_para(spin_groups_.size(), 0.0);
  std::vector<double> mtsq_perp(spin_groups_.size(), 0.0);

  std::vector<double> mjmt_para(spin_groups_.size(), 0.0);
  std::vector<double> mjmt_perp(spin_groups_.size(), 0.0);

  std::vector<double> mn(spin_groups_.size(), 0.0);

  for (std::size_t group_index = 0; group_index < spin_groups_.size(); ++group_index) {
    const auto& group = spin_groups_[group_index];
    // Uses the WMI geometry from M. Althammer,Phys. Rev. B 87, 224401 (2013).
    // assuming axes:
    // j -> x
    // t -> y
    // n -> z
    for (const auto i : group.indices_span()) {
      mtsq_para[group_index] +=  spins(i, 1) * spins(i, 1);
      mtsq_perp[group_index] +=  spins(i, 0) * spins(i, 0);
      mjmt_para[group_index] +=  spins(i, 0) * spins(i, 1);
      mjmt_perp[group_index] += -spins(i, 0) * spins(i, 1);

      mn[group_index] += spins(i, 2);
    }
  }

  for (std::size_t i = 0; i < spin_groups_.size(); ++i) {
    if (!spin_groups_[i].empty()) {
      const auto norm = 1.0 / static_cast<double>(spin_groups_[i].size());
      mtsq_para[i] = mtsq_para[i] * norm;
      mtsq_perp[i] = mtsq_perp[i] * norm;

      mjmt_para[i] = mjmt_para[i] * norm;
      mjmt_perp[i] = mjmt_perp[i] * norm;

      mn[i] = mn[i] * norm;
    }
  }

  std::vector<double> values;
  values.reserve(tsv_.num_cols());
  solver.append_monitor_coordinates(values);

  for (std::size_t i = 0; i < spin_groups_.size(); ++i) {
    values.push_back(mtsq_para[i]);
    values.push_back(mtsq_perp[i]);
    values.push_back(mjmt_para[i]);
    values.push_back(mjmt_perp[i]);
    values.push_back(mn[i]);
  }

  tsv_.write_row(values);
}

jams::output::TsvWriter SMRMonitor::make_tsv_writer() const {
  auto cols = globals::solver->monitor_coordinate_columns();

  for (const auto& group : spin_groups_) {
    for (const auto& component : {"mtsq_para", "mtsq_perp", "mjmt_para", "mjmt_perp", "mn"}) {
      cols.push_back({
          jams::monitors::grouped_column_name(grouping_, group.name, component),
          "dimensionless"});
    }
  }

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols),
      precision_);
}
