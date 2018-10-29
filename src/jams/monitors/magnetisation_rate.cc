// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include "jams/helpers/consts.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"

#include "magnetisation_rate.h"
using namespace std;

MagnetisationRateMonitor::MagnetisationRateMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(),
  magnetisation_stats_(),
  convergence_is_on_(false),              // do we want to use convergence in this monitor
  convergence_tolerance_(1.0),            // 1 standard deviation from the mean
  convergence_geweke_diagnostic_(100.0)   // number much larger than 1
{
  using namespace globals;

  if (settings.exists("convergence")) {
    convergence_is_on_ = true;
    convergence_tolerance_ = settings["convergence"];
    cout << "  convergence tolerance " << convergence_tolerance_ << "\n";
  }

  tsv_file.open(seedname + "_dm_dt.tsv");
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();

  material_count_.resize(lattice->num_materials(), 0);
  for (auto i = 0; i < num_spins; ++i) {
    material_count_[lattice->atom_material_id(i)]++;
  }
}

void MagnetisationRateMonitor::update(Solver * solver) {
  using namespace globals;

  std::vector<Vec3> dm_dt(::lattice->num_materials(), {0.0, 0.0, 0.0});

  for (auto i = 0; i < num_spins; ++i) {
    const auto type = lattice->atom_material_id(i);
    for (auto j = 0; j < 3; ++j) {
      dm_dt[type][j] += ds_dt(i, j);
    }
  }

  for (auto type = 0; type < lattice->num_materials(); ++type) {
    if (material_count_[type] == 0) continue;
    for (auto j = 0; j < 3; ++j) {
      dm_dt[type][j] /= static_cast<double>(material_count_[type]);
    }
  }

  tsv_file.width(12);
  tsv_file << std::scientific << solver->time() << "\t";

  for (auto type = 0; type < lattice->num_materials(); ++type) {
    for (auto j = 0; j < 3; ++j) {
      tsv_file << dm_dt[type][j] << "\t";
    }
  }

    if (convergence_is_on_) {
      double total_dm_dt = 0.0;
      for (auto type = 0; type < lattice->num_materials(); ++type) {
        total_dm_dt += abs(dm_dt[type]);
      }

      magnetisation_stats_.add(total_dm_dt);
      double nse = 0.0;
      magnetisation_stats_.geweke(convergence_geweke_diagnostic_, nse);
      tsv_file << convergence_geweke_diagnostic_;
    }

    tsv_file << std::endl;
}

bool MagnetisationRateMonitor::is_converged() {
  return ((std::abs(convergence_geweke_diagnostic_) < convergence_tolerance_) && convergence_is_on_);
}

std::string MagnetisationRateMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";

  for (auto i = 0; i < lattice->num_materials(); ++i) {
    auto name = lattice->material_name(i);
    ss << name + "_dmx_dt\t";
    ss << name + "_dmy_dt\t";
    ss << name + "_dmz_dt\t";
  }

  if (convergence_is_on_) {
    ss << "geweke_abs_dm_dt\t";
  }
  ss << std::endl;

  return ss.str();
}
