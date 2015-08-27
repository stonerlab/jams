// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation.h"

MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\ninitialising Magnetisation monitor");

  is_equilibration_monitor_ = true;

  std::string name = seedname + "_mag.dat";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << "#";
  outfile << std::setw(11) << "time";
  outfile << std::setw(16) << "temperature";
  outfile << std::setw(16) << "Hx";
  outfile << std::setw(16) << "Hy";
  outfile << std::setw(16) << "Hz";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(16) <<  lattice.material_name(i) + " -> " + "mx" ;
    outfile << std::setw(16) << "my";
    outfile << std::setw(16) << "mz";
    outfile << std::setw(16) << "|m|";
  }
  outfile << "\n";

  mag.resize(lattice.num_materials(), 4);
}

void MagnetisationMonitor::update(const Solver * const solver) {
  using namespace globals;

    int i, j;

    for (i = 0; i < lattice.num_materials(); ++i) {
      for (j = 0; j < 4; ++j) {
        mag(i, j) = 0.0;
      }
    }

    for (i = 0; i < num_spins; ++i) {
      int type = lattice.material(i);
      for (j = 0; j < 3; ++j) {
        mag(type, j) += s(i, j);
      }
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      for (j = 0; j < 3; ++j) {
        mag(i, j) = mag(i, j)/static_cast<double>(lattice.material_count(i));
      }
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      mag(i, 3) = sqrt(mag(i, 0)*mag(i, 0) + mag(i, 1)*mag(i, 1)
        + mag(i, 2)*mag(i, 2));
    }

    outfile << std::setw(12) << std::scientific << solver->time();
    outfile << std::setw(16) << std::fixed << solver->physics()->temperature();

    for (i = 0; i < 3; ++i) {
      outfile <<  std::setw(16) << solver->physics()->applied_field(i);
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      outfile << std::setw(16) << mag(i, 0);
      outfile << std::setw(16) << mag(i, 1);
      outfile << std::setw(16) << mag(i, 2);
      outfile << std::setw(16) << mag(i, 3);
    }

    outfile << std::endl;
}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
