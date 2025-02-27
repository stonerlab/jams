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
#include "jams/helpers/output.h"

TorqueMonitor::TorqueMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::output::full_path_filename("torq.tsv")),
  torque_stats_(),
  convergence_geweke_diagnostic_()
{
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void TorqueMonitor::update(Solver& solver) {
  tsv_file.width(12);
  tsv_file << std::scientific << solver.time() << "\t";

  // Loop over all of the Hamiltonians to calculate the total torque from each
  // Hamiltonian term. Each torque will be expressed as a torque per spin
  // and appended to a std::vector.
  std::vector<Vec3> torques;
  for (auto &hamiltonian : solver.hamiltonians()) {
    hamiltonian->calculate_fields(solver.time());

    // Loop over all spins in the system and sum the torque for the current
    // Hamiltonian
    Vec3 torque = {0.0, 0.0, 0.0};
    for (auto i = 0; i < globals::num_spins; ++i) {
      // Calculate the local torque on a lattice site (\vec{S} \times \vec{H})
      const Vec3 spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
      const Vec3 field = {hamiltonian->field(i, 0), hamiltonian->field(i, 1), hamiltonian->field(i, 2)};

      torque += cross(spin, field);
    }

    // In JAMS internal units energies are normalised by mu_B so we undo that
    // here
    torques.push_back(torque /static_cast<double>(globals::num_spins));
  }

  // Output all of the torques as columns in the tsv file
  for (const auto& torque : torques) {
    for (auto n = 0; n < 3; ++n) {
      tsv_file << std::scientific << torque[n] << "\t";
    }
  }

  if (convergence_status_ != Monitor::ConvergenceStatus::kDisabled && solver.time() > convergence_burn_time_) {
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

Monitor::ConvergenceStatus TorqueMonitor::convergence_status() {
  if (convergence_status_ == ConvergenceStatus::kDisabled) {
    return convergence_status_;
  }

  if (!convergence_geweke_diagnostic_.empty()) {
    for (double &x : convergence_geweke_diagnostic_) {
      if (std::abs(x) > convergence_tolerance_) {
        convergence_status_ = ConvergenceStatus::kNotConverged;
        return convergence_status_;
      }
    }
  }

  // if we made it through the loop without returning then all elements of
  // convergence_geweke_diagnostic_ must be converged
  convergence_status_ = ConvergenceStatus::kConverged;
  return convergence_status_;
}

std::string TorqueMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  for (auto &hamiltonian : globals::solver->hamiltonians()) {
    const auto name = hamiltonian->name();
    ss << name + "_tx\t";
    ss << name + "_ty\t";
    ss << name + "_tz\t";

    if (convergence_status_ != ConvergenceStatus::kDisabled) {
      tsv_file << name + "_geweke";
    }
  }

  ss << std::endl;
  return ss.str();
}
