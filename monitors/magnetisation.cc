// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation.h"

MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  mag(::lattice.num_materials(), 4),
  s_transform_(globals::num_spins, 3),
  outfile(),
  m2_stats_(),
  m4_stats_(),
  convergence_is_on_(false),              // do we want to use convergence in this monitor
  convergence_tolerance_(1.0),            // 1 standard deviation from the mean
  convergence_geweke_m2_diagnostic_(100.0),   // number much larger than 1
  convergence_geweke_m4_diagnostic_(100.0)   // number much larger than 1
{
  using namespace globals;
  ::output.write("\ninitialising Magnetisation monitor\n");

  if (settings.exists("convergence")) {
    convergence_is_on_ = true;
    convergence_tolerance_ = settings["convergence"];
    ::output.write("  convergence tolerance: %f\n", convergence_tolerance_);
  }

  is_equilibration_monitor_ = true;

  // create transform arrays for example to apply a Holstein Primakoff transform
  s_transform_.resize(num_spins, 3);

  libconfig::Setting& material_settings = ::config.lookup("materials");
  for (int i = 0; i < num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      s_transform_(i,n) = material_settings[::lattice.atom_material(i)]["transform"][n];
    }
  }

  std::string name = seedname + "_mag.tsv";
  outfile.open(name.c_str());
  outfile.setf(std::ios::right);

  // header for the magnetisation file
  outfile << std::setw(12) << "time" << "\t";
  outfile << std::setw(12) << "temperature" << "\t";
  outfile << std::setw(12) << "Hx" << "\t";
  outfile << std::setw(12) << "Hy" << "\t";
  outfile << std::setw(12) << "Hz" << "\t";

  for (int i = 0; i < lattice.num_materials(); ++i) {
    outfile << std::setw(12) << lattice.material_name(i) + ":mx" << "\t";
    outfile << std::setw(12) << lattice.material_name(i) + ":my" << "\t";
    outfile << std::setw(12) << lattice.material_name(i) + ":mz" << "\t";
    outfile << std::setw(12) << lattice.material_name(i) + ":m" << "\t";
  }

  outfile << std::setw(12) << "m2" << "\t";
  outfile << std::setw(12) << "m4" << "\t";
  outfile << std::setw(12) << "binder" << "\t";

  if (convergence_is_on_) {
    outfile << std::setw(12) << "geweke:m2" << "\t";
    outfile << std::setw(12) << "geweke:m4";
  }

  outfile << "\n";
}

void MagnetisationMonitor::update(Solver * solver) {
  using namespace globals;

    int i, j;

    mag.zero();

    for (i = 0; i < num_spins; ++i) {
      int type = lattice.atom_material(i);
      for (j = 0; j < 3; ++j) {
        mag(type, j) += s(i, j);
      }
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      for (j = 0; j < 3; ++j) {
        mag(i, j) = mag(i, j)/static_cast<double>(lattice.num_of_material(i));
      }
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      mag(i, 3) = sqrt(mag(i, 0)*mag(i, 0) + mag(i, 1)*mag(i, 1)
        + mag(i, 2)*mag(i, 2));
    }

    outfile << std::setw(12) << std::scientific << solver->time() << "\t";
    outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";

    for (i = 0; i < 3; ++i) {
      outfile <<  std::setw(12) << solver->physics()->applied_field(i) << "\t";
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      outfile << std::setw(12) << mag(i, 0) << "\t";
      outfile << std::setw(12) << mag(i, 1) << "\t";
      outfile << std::setw(12) << mag(i, 2) << "\t";
      outfile << std::setw(12) << mag(i, 3) << "\t";
    }

    double m2 = binder_m2();
    m2_stats_.add(m2);
    m4_stats_.add(m2 * m2);


    outfile << std::setw(12) << m2 << "\t";
    outfile << std::setw(12) << m2 * m2 << "\t";
    outfile << std::setw(12) << binder_cumulant() << "\t";

    if (convergence_is_on_) {
      convergence_geweke_m2_diagnostic_ = m2_stats_.geweke();
      convergence_geweke_m4_diagnostic_ = m4_stats_.geweke();
      outfile << std::setw(12) << convergence_geweke_m2_diagnostic_;
      outfile << std::setw(12) << convergence_geweke_m4_diagnostic_;
    }

    outfile << std::endl;
}

double MagnetisationMonitor::binder_m2() {
  using namespace globals;

  jblib::Vec3<double> m;

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      m[j] = m[j] + s_transform_(i, j) * s(i, j);
    }
  }

  return m.norm_sq() / square(static_cast<double>(num_spins));
}

double MagnetisationMonitor::binder_cumulant() {
  return 1.0 - (m4_stats_.mean()) / (3.0 * square(m2_stats_.mean()));
}

bool MagnetisationMonitor::is_converged() {
  return ((std::abs(convergence_geweke_m2_diagnostic_) < convergence_tolerance_) &&(std::abs(convergence_geweke_m4_diagnostic_) < convergence_tolerance_) && convergence_is_on_);
}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
