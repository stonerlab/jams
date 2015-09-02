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
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Torque monitor...\n");

  is_equilibration_monitor_ = true;

  std::string name = seedname + "_torq.dat";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << "#";
  outfile << std::setw(11) << "time";
  outfile << std::setw(16) << "temperature";
  outfile << std::setw(16) << "Tx";
  outfile << std::setw(16) << "Ty";
  outfile << std::setw(16) << "Tz";
  outfile << "\n";
}

void TorqueMonitor::update(Solver * solver) {
  using namespace globals;

    int i;
    jblib::Vec3<double> total_torque(0,0,0);

    outfile << std::setw(12) << std::scientific << solver->time();
    outfile << std::setw(16) << std::scientific << solver->physics()->temperature();

    for (std::vector<Hamiltonian*>::iterator it = solver->hamiltonians().begin() ; it != solver->hamiltonians().end(); ++it) {
      torque_.x = 0; torque_.y = 0; torque_.z = 0;

      (*it)->calculate_fields();

      for (i = 0; i < num_spins; ++i) {
        torque_[0] += s(i,1)*(*it)->field(i,2) - s(i,2)*(*it)->field(i,1);
        torque_[1] += s(i,2)*(*it)->field(i,0) - s(i,0)*(*it)->field(i,2);
        torque_[2] += s(i,0)*(*it)->field(i,1) - s(i,1)*(*it)->field(i,0);
      }

      for (i = 0; i < 3; ++i) {
        outfile <<  std::setw(16) << torque_[i]*kBohrMagneton/static_cast<double>(num_spins);
      }

      for (i = 0; i < 3; ++i) {
        total_torque[i] += torque_[i]*kBohrMagneton/static_cast<double>(num_spins);
      }
    }

    torque_stats_.add(abs(total_torque));

    outfile << std::setw(16) << torque_stats_.geweke();

    outfile << std::endl;
}

TorqueMonitor::~TorqueMonitor() {
  outfile.close();
}
