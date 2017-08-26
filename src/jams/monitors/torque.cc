// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <vector>

#include "jams/core/output.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/stats.h"
#include "jams/core/consts.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/monitors/torque.h"

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"

TorqueMonitor::TorqueMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  outfile(),
  torque_stats_(),
  convergence_geweke_diagnostic_()
{
  using namespace globals;
  ::output->write("\nInitialising Torque monitor...\n");
}

void TorqueMonitor::update(Solver * solver) {
  using namespace globals;

  static bool first_run = true;

  if (first_run) {
    open_outfile();
    first_run = false;
  }

  for (int n = 0; n < 3; ++n) {
    convergence_geweke_diagnostic_[n] = 100.0; // number much larger than 1
  }

  int i;
  const double norm = kBohrMagneton/static_cast<double>(num_spins);

  outfile << std::setw(12) << std::scientific << solver->time() << "\t";
  outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";

  jblib::Vec3<double> total_torque(0.0, 0.0, 0.0);
  // output torque from each hamiltonian term
  for (std::vector<Hamiltonian*>::iterator it = solver->hamiltonians().begin(); it != solver->hamiltonians().end(); ++it) {
    jblib::Vec3<double> this_torque(0.0, 0.0, 0.0);

    (*it)->calculate_fields();

    for (i = 0; i < num_spins; ++i) {
      this_torque[0] += s(i,1)*(*it)->field(i,2) - s(i,2)*(*it)->field(i,1);
      this_torque[1] += s(i,2)*(*it)->field(i,0) - s(i,0)*(*it)->field(i,2);
      this_torque[2] += s(i,0)*(*it)->field(i,1) - s(i,1)*(*it)->field(i,0);
    }

    total_torque = total_torque + this_torque * norm;

    for (i = 0; i < 3; ++i) {
      outfile <<  std::setw(12) << std::scientific << this_torque[i] * norm << "\t";
    }
  }
  
  // convergence stats

  if (convergence_is_on_ && solver->time() > convergence_burn_time_) {
    for (int n = 0; n < 3; ++n) {
      torque_stats_[n].add(total_torque[n]);
      torque_stats_[n].geweke(convergence_geweke_diagnostic_[n], convergence_stderr_);

      if (torque_stats_[n].size() > 1 && torque_stats_[n].size() % 10 == 0) {
        outfile << std::setw(12) << convergence_geweke_diagnostic_[n] << "\t";
      } else {
        outfile << std::setw(12) << "--------";
      }
    }
  }

  outfile << std::endl;
}

void TorqueMonitor::open_outfile() {
  std::string name = seedname + "_torq.tsv";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // column headers
  outfile << std::setw(12) << "time" << "\t";
  outfile << std::setw(12) << "temperature" << "\t";

  for (std::vector<Hamiltonian*>::iterator it = solver->hamiltonians().begin() ; it != solver->hamiltonians().end(); ++it) {
    outfile << std::setw(12) << (*it)->name()+":Tx" << "\t";
    outfile << std::setw(12) << (*it)->name()+":Ty" << "\t";
    outfile << std::setw(12) << (*it)->name()+":Tz" << "\t";
    outfile << std::setw(12) << (*it)->name()+":geweke";
  }
  outfile << "\n";
}

bool TorqueMonitor::is_converged() {

  if (convergence_is_on_ && convergence_geweke_diagnostic_.size() > 0) {
    for (auto it = convergence_geweke_diagnostic_.begin() ; it != convergence_geweke_diagnostic_.end(); ++it) {
      if (std::abs(*it) > convergence_tolerance_) {
        return false;
      }
    }

    return true;
  }

  return false;
}

TorqueMonitor::~TorqueMonitor() {
  if (outfile.is_open()) {
    outfile.close();
  }
}
