// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <vector>

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/stats.h"
#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/monitors/torque.h"
#include "jams/containers/vec3.h"

TorqueMonitor::TorqueMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::filesystem::open_file(seedname + "_torq.tsv")),
  torque_stats_(),
  convergence_geweke_diagnostic_()
{
  using namespace globals;

  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void TorqueMonitor::update(Solver * solver) {
  using namespace globals;

  tsv_file.width(12);
  tsv_file << std::scientific << solver->time() << "\t";

  std::vector<Vec3> torques;
  for (auto &hamiltonian : solver->hamiltonians()) {
    hamiltonian->calculate_fields();

    Vec3 torque = {0.0, 0.0, 0.0};
    for (auto i = 0; i < num_spins; ++i) {
      const Vec3 spin = {s(i,0), s(i,1), s(i,2)};
      const Vec3 field = {hamiltonian->field(i, 0), hamiltonian->field(i, 1), hamiltonian->field(i, 2)};
      torque += cross(spin, field);
    }

    torques.push_back(torque * kBohrMagneton /static_cast<double>(num_spins));
  }

  for (const auto& torque : torques) {
    for (auto n = 0; n < 3; ++n) {
      tsv_file << std::scientific << torque[n] << "\t";
    }
  }

  if (convergence_is_on_ && solver->time() > convergence_burn_time_) {
    convergence_geweke_diagnostic_ = {100.0, 100.0, 100.0}; // number much larger than 1

    Vec3 total_torque = {0.0, 0.0, 0.0};
    for (const auto& torque : torques) {
      total_torque += torque;
    }

    for (auto n = 0; n < 3; ++n) {
      torque_stats_[n].add(total_torque[n]);
      torque_stats_[n].geweke(convergence_geweke_diagnostic_[n], convergence_stderr_);

      if (torque_stats_[n].size() > 1 && torque_stats_[n].size() % 10 == 0) {
        tsv_file << convergence_geweke_diagnostic_[n] << "\t";
      } else {
        tsv_file << "--------";
      }
    }
  }

  tsv_file << std::endl;
}

bool TorqueMonitor::is_converged() {
  if (convergence_is_on_ && !convergence_geweke_diagnostic_.empty()) {
    for (double &x : convergence_geweke_diagnostic_) {
      if (std::abs(x) > convergence_tolerance_) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::string TorqueMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  for (auto &hamiltonian : solver->hamiltonians()) {
    const auto name = hamiltonian->name();
    ss << name + "_tx\t";
    ss << name + "_ty\t";
    ss << name + "_tz\t";

    if (convergence_is_on_) {
      tsv_file << name + "_geweke";
    }
  }

  ss << std::endl;
  return ss.str();
}
