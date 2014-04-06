// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/consts.h"
#include "core/globals.h"
#include "core/lattice.h"

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

void TorqueMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

    int i, j;

    torque_.x = 0; torque_.y = 0; torque_.z = 0;

    for (i = 0; i < num_spins; ++i) {
      torque_[0] += s(i,1)*(h(i,2) + h_dipole(i,2)) - s(i,2)*(h(i,1) + h_dipole(i,1));
      torque_[1] += s(i,2)*(h(i,0) + h_dipole(i,0)) - s(i,0)*(h(i,2) + h_dipole(i,2));
      torque_[2] += s(i,0)*(h(i,1) + h_dipole(i,1)) - s(i,1)*(h(i,0) + h_dipole(i,0));
    }

    for (j = 0; j < 3; ++j) {
      torque_[j] = torque_[j]*mu_bohr_si/static_cast<double>(num_spins);
    }

    outfile << std::setw(12) << std::scientific << time;
    outfile << std::setw(16) << std::scientific << temperature;

    for (i = 0; i < 3; ++i) {
      outfile <<  std::setw(16) << torque_[i];
    }

    outfile << "\n";
}

TorqueMonitor::~TorqueMonitor() {
  outfile.close();
}
