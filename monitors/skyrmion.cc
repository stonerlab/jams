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

  if (settings.exists("thresholds")) {
    for (int n = 0; n < settings["thresholds"].getLength(); ++n) {
      thresholds.push_back(settings["thresholds"][n]);
    }
  } else {
    thresholds.push_back(0.0);
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
    for (int j = 0; j < thresholds.size(); ++j) {
      outfile << std::setw(16) << "R_g" << "[t" << thresholds[j] << "]";
      outfile << std::setw(16) << "2R_g/sqrt(2)" << "[t" << thresholds[j] << "]";
    }
  }
  outfile << "\n";
}

void SkyrmionMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

    int i, n;

    outfile << std::setw(12) << std::scientific << time;
    outfile << std::setw(16) << std::fixed << temperature;

    // TODO: account for periodic bounaries

    for (int t = 0; t < thresholds.size(); ++t) {

      std::vector<jblib::Vec3<double> > r_avg(lattice.num_materials(), jblib::Vec3<double>(0.0, 0.0, 0.0));
      std::vector<double> r_sq(lattice.num_materials(), 0.0);
      std::vector<int> r_count(lattice.num_materials(), 0);

      for (int i = 0; i < num_spins; ++i) {
        int type = lattice.get_material_number(i);
        if (s(i, 2)*type_norms[type] > thresholds[t]) {
          r_avg[type] += lattice.lattice_positions_[i];
          r_sq[type] += dot(lattice.lattice_positions_[i], lattice.lattice_positions_[i]);
          r_count[type]++;
        }
      }

      for (n = 0; n < lattice.num_materials(); ++n) {
        if (r_count[n] == 0) {
          for (i = 0; i < 5; ++i) {
            outfile << std::setw(16) << 0.0;
          }
        } else {
          r_avg[n] /= static_cast<double>(r_count[n]);
          r_sq[n] /= static_cast<double>(r_count[n]);
          for (i = 0; i < 3; ++i) {
            outfile << std::setw(16) << lattice.lattice_parameter_*r_avg[n][i];
          }
          double r_gyration = lattice.lattice_parameter_*sqrt(r_sq[n]-dot(r_avg[n], r_avg[n]));
          outfile << std::setw(16) << r_gyration << std::setw(16) << (2.0/sqrt(2.0))*r_gyration;
        }
      }
    }

    outfile << "\n";

}

SkyrmionMonitor::~SkyrmionMonitor() {
  outfile.close();
}
