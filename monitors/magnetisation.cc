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
  outfile(),
  mx_stats_(),
  mz_stats_(),
  convergence_is_on_(false),              // do we want to use convergence in this monitor
  convergence_tolerance_(1.0),            // 1 standard deviation from the mean
  convergence_geweke_mx_diagnostic_(100.0),   // number much larger than 1
  convergence_geweke_mz_diagnostic_(100.0)   // number much larger than 1
{
  using namespace globals;
  ::output.write("\ninitialising Magnetisation monitor\n");

  if (settings.exists("convergence")) {
    convergence_is_on_ = true;
    convergence_tolerance_ = settings["convergence"];
    ::output.write("  convergence tolerance: %f\n", convergence_tolerance_);
  }

  is_equilibration_monitor_ = true;

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

  if (convergence_is_on_) {
    outfile << std::setw(12) << "geweke";
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

    if (convergence_is_on_) {
      mz_stats_.add(mag(0, 2));
      mx_stats_.add(mag(0, 0));

      convergence_geweke_mx_diagnostic_ = mx_stats_.geweke();
      convergence_geweke_mz_diagnostic_ = mz_stats_.geweke();
      outfile << std::setw(12) << convergence_geweke_mx_diagnostic_;
      outfile << std::setw(12) << convergence_geweke_mz_diagnostic_;
    }

    outfile << std::endl;
}

bool MagnetisationMonitor::is_converged() {
  return ((std::abs(convergence_geweke_mz_diagnostic_) < convergence_tolerance_) &&(std::abs(convergence_geweke_mx_diagnostic_) < convergence_tolerance_) && convergence_is_on_);
}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
