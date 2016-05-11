// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/consts.h"
#include "core/globals.h"
#include "core/lattice.h"
#include "core/hamiltonian.h"

#include "monitors/torque.h"

TorqueMonitor::TorqueMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  outfile(),
  torque_stats_(),
  convergence_is_on_(false),              // do we want to use convergence in this monitor
  convergence_tolerance_(1.0),            // 1 standard deviation from the mean
  convergence_geweke_diagnostic_()   // number much larger than 1
{
  using namespace globals;
  ::output.write("\nInitialising Torque monitor...\n");

  if (settings.exists("convergence")) {
    convergence_is_on_ = true;
    convergence_tolerance_ = settings["convergence"];
  }

  is_equilibration_monitor_ = true;


}

void TorqueMonitor::update(Solver * solver) {
  using namespace globals;

  static bool first_run = true;

  if (first_run) {
    open_outfile();
    first_run = false;
  }

  int i;
  jblib::Vec3<double> torque;
  jblib::Vec3<double> total_torque(0,0,0);
  const double norm = kBohrMagneton/static_cast<double>(num_spins);

  outfile << std::setw(12) << std::scientific << solver->time() << "\t";
  outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";

  if (torque_stats_.size() != solver->hamiltonians().size()){
    torque_stats_.resize(solver->hamiltonians().size());
  }

  if (convergence_geweke_diagnostic_.size() != solver->hamiltonians().size()){
    convergence_geweke_diagnostic_.resize(solver->hamiltonians().size());
  }

  int count = 0;
  for (std::vector<Hamiltonian*>::iterator it = solver->hamiltonians().begin(); it != solver->hamiltonians().end(); ++it) {
    torque.x = 0; torque.y = 0; torque.z = 0;

    (*it)->calculate_fields();

    for (i = 0; i < num_spins; ++i) {
      torque[0] += s(i,1)*(*it)->field(i,2) - s(i,2)*(*it)->field(i,1);
      torque[1] += s(i,2)*(*it)->field(i,0) - s(i,0)*(*it)->field(i,2);
      torque[2] += s(i,0)*(*it)->field(i,1) - s(i,1)*(*it)->field(i,0);
    }

    for (i = 0; i < 3; ++i) {
      outfile <<  std::setw(12) << std::scientific << torque[i] * norm << "\t";
    }

    torque_stats_[count].add(torque[1]/static_cast<double>(num_spins));
    convergence_geweke_diagnostic_[count] = torque_stats_[count].geweke();
    count++;
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
