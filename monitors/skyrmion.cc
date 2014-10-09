// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <vector>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/skyrmion.h"

SkyrmionMonitor::SkyrmionMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Skyrmion monitor...\n");

  is_equilibration_monitor_ = true;

  type_norms.resize(lattice.num_materials(), 1.0);

  if (settings.exists("type_norms")) {
    for (int n = 0; n < lattice.num_materials(); ++n) {
      type_norms[n] = settings["type_norms"][n];
    }
  }

  std::string name = seedname + "_sky.dat";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << "#";
  outfile << std::setw(11) << "time";
  outfile << std::setw(16) << "temperature";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(16) <<  lattice.get_material_name(i) + " -> " + "r_avg(x,y,z)";
    outfile << std::setw(16) << "R_gyration";
    outfile << std::setw(16) << "(2/sqrt(2))*R_gyration";
  }
  outfile << "\n";
}

void SkyrmionMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

    int i, n;

    outfile << std::setw(12) << std::scientific << time;
    outfile << std::setw(16) << std::fixed << temperature;

    // TODO: account for periodic bounaries

    std::vector<jblib::Vec3<double> > r_avg(lattice.num_materials(), jblib::Vec3<double>(0.0, 0.0, 0.0));
    std::vector<double> r_sq(lattice.num_materials(), 0.0);
    std::vector<int> r_count(lattice.num_materials(), 0);

    for (n = 0; n < lattice.num_materials(); ++n) {
      for (int i = 0; i < num_spins; ++i) {
        if (s(i, 2)*type_norms[n] > 0.0) {
          r_avg[n] += lattice.lattice_positions_[i];
          r_sq[n] += dot(lattice.lattice_positions_[i], lattice.lattice_positions_[i]);
          r_count[n]++;
        }
      }
    }

    for (n = 0; n < lattice.num_materials(); ++n) {
      r_avg[n] /= static_cast<double>(r_count[n]);
      r_sq[n] /= static_cast<double>(r_count[n]);
      for (i = 0; i < 3; ++i) {
        outfile << std::setw(16) << lattice.lattice_parameter_*r_avg[n][i];
      }
      double r_gyration = lattice.lattice_parameter_*sqrt(r_sq[n]-dot(r_avg[n], r_avg[n]));
      outfile << std::setw(16) << r_gyration << std::setw(16) << (2.0/sqrt(2.0))*r_gyration;
    }

    outfile << std::endl;
}

SkyrmionMonitor::~SkyrmionMonitor() {
  outfile.close();
}
