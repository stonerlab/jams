// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation.h"

MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Magnetisation monitor...\n");

  is_equilibration_monitor_ = true;

  std::string name = seedname + "_mag.dat";
  outfile.open(name.c_str());

  // mx my mz |m|
  mag.resize(lattice.numTypes(), 4);
}

void MagnetisationMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

  if (iteration%output_step_freq_ == 0) {
    int i, j;

    for (i = 0; i < lattice.numTypes(); ++i) {
      for (j = 0; j < 4; ++j) {
        mag(i, j) = 0.0;
      }
    }

    for (i = 0; i < num_spins; ++i) {
      int type = lattice.getType(i);
      for (j = 0; j < 3; ++j) {
        mag(type, j) += s(i, j);
      }
    }

    for (i = 0; i < lattice.numTypes(); ++i) {
      for (j = 0; j < 3; ++j) {
        mag(i, j) = mag(i, j)/static_cast<double>(lattice.getTypeCount(i));
      }
    }

    for (i = 0; i < lattice.numTypes(); ++i) {
      mag(i, 3) = sqrt(mag(i, 0)*mag(i, 0) + mag(i, 1)*mag(i, 1)
        + mag(i, 2)*mag(i, 2));
    }

    outfile << time;

    outfile << "\t" << temperature;

    for (i = 0; i < 3; ++i) {
      outfile << "\t" << applied_field[i];
    }

    for (i = 0; i < lattice.numTypes(); ++i) {
      outfile <<"\t"<< mag(i, 0) <<"\t"<< mag(i, 1) <<"\t"<< mag(i, 2)
      <<"\t"<< mag(i, 3);
    }

    outfile << "\n";
  }
}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
