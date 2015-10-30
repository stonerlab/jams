// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include "core/consts.h"

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation_rate.h"

MagnetisationRateMonitor::MagnetisationRateMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  dm_dt(::lattice.num_materials(), 4),
  outfile(),
  magnetisation_stats_(),
  convergence_is_on_(false),              // do we want to use convergence in this monitor
  convergence_tolerance_(1.0),            // 1 standard deviation from the mean
  convergence_geweke_diagnostic_(100.0)   // number much larger than 1
{
  using namespace globals;
  ::output.write("\ninitialising MagnetisationRate monitor\n");

  if (settings.exists("convergence")) {
    convergence_is_on_ = true;
    convergence_tolerance_ = settings["convergence"];
    ::output.write("  convergence tolerance: %f\n", convergence_tolerance_);
  }

  is_equilibration_monitor_ = true;

  std::string name = seedname + "_dm_dt.tsv";
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

void MagnetisationRateMonitor::update(Solver * solver) {
  using namespace globals;

    int i, j;

    dm_dt.zero();

    for (i = 0; i < num_spins; ++i) {
      int type = lattice.material(i);
      for (j = 0; j < 3; ++j) {
        dm_dt(type, j) += ds_dt(i, j);
      }
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      for (j = 0; j < 3; ++j) {
        dm_dt(i, j) = dm_dt(i, j)/static_cast<double>(lattice.material_count(i));
      }
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      dm_dt(i, 3) = sqrt(dm_dt(i, 0)*dm_dt(i, 0) + dm_dt(i, 1)*dm_dt(i, 1)
        + dm_dt(i, 2)*dm_dt(i, 2));
    }

    outfile << std::setw(12) << std::scientific << solver->time() << "\t";
    outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";

    for (i = 0; i < 3; ++i) {
      outfile <<  std::setw(12) << solver->physics()->applied_field(i) << "\t";
    }

    for (i = 0; i < lattice.num_materials(); ++i) {
      outfile << std::setw(12) << std::scientific << dm_dt(i, 0)*kGyromagneticRatio << "\t";
      outfile << std::setw(12) << std::scientific << dm_dt(i, 1)*kGyromagneticRatio << "\t";
      outfile << std::setw(12) << std::scientific << dm_dt(i, 2)*kGyromagneticRatio << "\t";
      outfile << std::setw(12) << std::scientific << dm_dt(i, 3)*kGyromagneticRatio << "\t";
    }

    if (convergence_is_on_) {
      double total_dm_dt = 0.0;
      for (i = 0; i < lattice.num_materials(); ++i) {
        total_dm_dt += dm_dt(i, 3);
      }

      magnetisation_stats_.add(total_dm_dt);
      convergence_geweke_diagnostic_ = magnetisation_stats_.geweke();
      outfile << std::setw(12) << convergence_geweke_diagnostic_;
    }

    outfile << std::endl;
}

bool MagnetisationRateMonitor::is_converged() {
  return ((std::abs(convergence_geweke_diagnostic_) < convergence_tolerance_) && convergence_is_on_);
}

MagnetisationRateMonitor::~MagnetisationRateMonitor() {
  outfile.close();
}
