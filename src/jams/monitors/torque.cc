// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/monitors/torque.h"
#include "jams/helpers/output.h"

TorqueMonitor::TorqueMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  grouping_(jams::monitors::parse_spin_grouping(settings, "none", "torque")),
  spin_groups_(jams::monitors::make_spin_groups(grouping_)),
  precision_(jams::config_optional<int>(settings, "precision", 8)),
  torque_stats_(),
  convergence_geweke_diagnostic_(),
  tsv_(make_tsv_writer())
{
  torque_stats_.resize(spin_groups_.size());
  convergence_geweke_diagnostic_.resize(spin_groups_.size(), {100.0, 100.0, 100.0});
}

void TorqueMonitor::update(Solver& solver) {
  const auto torques = calculate_torques(solver);
  std::vector<std::optional<double>> values;
  values.reserve(tsv_.num_cols());

  values.push_back(solver.time());

  for (const auto& group_torques : torques) {
    for (const auto& torque : group_torques) {
      for (auto n = 0; n < 3; ++n) {
        values.push_back(torque[n]);
      }
    }
  }

  if (convergence_status_ != Monitor::ConvergenceStatus::kDisabled) {
    if (solver.time() > convergence_burn_time_) {
      for (std::size_t group_index = 0; group_index < torques.size(); ++group_index) {
        convergence_geweke_diagnostic_[group_index] = {100.0, 100.0, 100.0}; // number much larger than 1
        const auto total_torque = total_group_torque(torques[group_index]);

        for (auto n = 0; n < 3; ++n) {
          torque_stats_[group_index][n].add(total_torque[n]);
          torque_stats_[group_index][n].geweke(
              convergence_geweke_diagnostic_[group_index][n],
              convergence_stderr_);

          if (torque_stats_[group_index][n].size() > 1 && torque_stats_[group_index][n].size() % 10 == 0) {
            values.push_back(convergence_geweke_diagnostic_[group_index][n]);
          } else {
            values.push_back(std::nullopt);
          }
        }
      }
    } else {
      for (std::size_t group_index = 0; group_index < torques.size(); ++group_index) {
        for (auto n = 0; n < 3; ++n) {
          values.push_back(std::nullopt);
        }
      }
    }
  }

  tsv_.write_row(values);
}

TorqueMonitor::GroupedTorques TorqueMonitor::calculate_torques(Solver& solver) {
  const auto spins = globals::s.host_view();
  const auto num_groups = spin_groups_.size();
  const auto num_hamiltonians = solver.hamiltonians().size();
  GroupedTorques torques(num_groups, HamiltonianTorques(num_hamiltonians, {0.0, 0.0, 0.0}));

  for (std::size_t hamiltonian_index = 0; hamiltonian_index < num_hamiltonians; ++hamiltonian_index) {
    auto& hamiltonian = solver.hamiltonians()[hamiltonian_index];
    hamiltonian->calculate_fields(solver.time());

    for (std::size_t group_index = 0; group_index < num_groups; ++group_index) {
      const auto& group = spin_groups_[group_index];

      TorqueComponents torque = {0.0, 0.0, 0.0};
      for (const auto spin_index : group.indices_span()) {
        // Calculate the local torque on a lattice site (\vec{S} \times \vec{H})
        const TorqueComponents spin = {
            spins(spin_index, 0),
            spins(spin_index, 1),
            spins(spin_index, 2)};
        const TorqueComponents field = {
            hamiltonian->field(spin_index, 0),
            hamiltonian->field(spin_index, 1),
            hamiltonian->field(spin_index, 2)};

        torque += jams::cross(spin, field);
      }

      if (!group.empty()) {
        torque = torque / static_cast<double>(group.size());
      }

      torques[group_index][hamiltonian_index] = torque;
    }
  }

  return torques;
}

TorqueMonitor::TorqueComponents TorqueMonitor::total_group_torque(
    const HamiltonianTorques& torques) const {
  TorqueComponents total_torque = {0.0, 0.0, 0.0};
  for (const auto& torque : torques) {
    total_torque += torque;
  }
  return total_torque;
}

Monitor::ConvergenceStatus TorqueMonitor::convergence_status() {
  if (convergence_status_ == ConvergenceStatus::kDisabled) {
    return convergence_status_;
  }

  if (!convergence_geweke_diagnostic_.empty()) {
    for (auto &diagnostic : convergence_geweke_diagnostic_) {
      for (double &x : diagnostic) {
        if (std::abs(x) > convergence_tolerance_) {
          convergence_status_ = ConvergenceStatus::kNotConverged;
          return convergence_status_;
        }
      }
    }
  }

  // if we made it through the loop without returning then all elements of
  // convergence_geweke_diagnostic_ must be converged
  convergence_status_ = ConvergenceStatus::kConverged;
  return convergence_status_;
}

jams::output::TsvWriter TorqueMonitor::make_tsv_writer() const {
  std::vector<jams::output::ColDef> cols;
  cols.push_back({"time", "picoseconds"});

  for (const auto& group : spin_groups_) {
    for (const auto& hamiltonian : globals::solver->hamiltonians()) {
      const auto name = hamiltonian->name();
      cols.push_back({torque_column_name(group, name, "tx"), "meV"});
      cols.push_back({torque_column_name(group, name, "ty"), "meV"});
      cols.push_back({torque_column_name(group, name, "tz"), "meV"});
    }
  }

  if (convergence_status_ != ConvergenceStatus::kDisabled) {
    for (const auto& group : spin_groups_) {
      cols.push_back({convergence_column_name(group, "tx"), "dimensionless"});
      cols.push_back({convergence_column_name(group, "ty"), "dimensionless"});
      cols.push_back({convergence_column_name(group, "tz"), "dimensionless"});
    }
  }

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols),
      precision_);
}

std::string TorqueMonitor::torque_column_name(
    const jams::monitors::SpinGroup& group,
    const std::string& hamiltonian_name,
    const std::string& component) const {
  return jams::monitors::grouped_column_name(
      grouping_,
      group.name,
      hamiltonian_name + "_" + component);
}

std::string TorqueMonitor::convergence_column_name(
    const jams::monitors::SpinGroup& group,
    const std::string& component) const {
  return jams::monitors::grouped_column_name(
      grouping_,
      group.name,
      component + "_geweke");
}
