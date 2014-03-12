// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/consts.h"

#include "monitors/anisotropy_energy.h"

AnisotropyEnergyMonitor::AnisotropyEnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Anisotropy Energy monitor...\n");

  is_equilibration_monitor_ = true;

  std::string name = seedname + "_dz.dat";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << "#";
  outfile << std::setw(11) << "time";
  outfile << std::setw(16) << "temperature";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(16) <<  lattice.get_material_name(i) + " -> " + "d2z";
    outfile << std::setw(16) << "d4z";
    outfile << std::setw(16) << "d6z";
  }
  outfile << "\n";

  dz_energy_.resize(lattice.num_materials(), 3);
}

void AnisotropyEnergyMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;


    int i, j;

    for (i = 0; i < lattice.num_materials(); ++i) {
      for (j = 0; j < 3; ++j) {
        dz_energy_(i, j) = 0.0;
      }
    }

    for (i = 0; i < num_spins; ++i) {
      int type = lattice.get_material_number(i);
      // factors of 1/2 and 1/8 are removed so we are calculating d2z and d4z directly
      dz_energy_(type, 0) += -d2z(i)*0.5*(3.0*s(i, 2)*s(i, 2) - 1.0);
      dz_energy_(type, 1) += -d4z(i)*0.125*(35.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)-30.0*s(i, 2)*s(i, 2) + 3.0);
      dz_energy_(type, 2) += -d6z(i)*0.0625*(231.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) - 315.0*s(i, 2)*s(i, 2)*s(i, 2)*s(i, 2) + 105.0*s(i, 2)*s(i, 2) - 5.0);
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      for (j = 0; j < 3; ++j) {
        dz_energy_(i, j) = dz_energy_(i, j)/static_cast<double>(lattice.num_spins_of_material(i));
      }
    }

    outfile << std::scientific << std::setw(12) << time;
    outfile << std::fixed << std::setw(16) << temperature;

    for (i = 0; i < lattice.num_materials(); ++i) {
      outfile << std::scientific << std::setw(16) << dz_energy_(i, 0)*mu_bohr_si;
      outfile << std::scientific << std::setw(16) << dz_energy_(i, 1)*mu_bohr_si;
      outfile << std::scientific << std::setw(16) << dz_energy_(i, 2)*mu_bohr_si;
    }
    outfile << "\n";
}

AnisotropyEnergyMonitor::~AnisotropyEnergyMonitor() {
  outfile.close();
}
