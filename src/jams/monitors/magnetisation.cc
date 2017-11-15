// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>

#include "jams/core/output.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/maths.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"

#include "jams/monitors/magnetisation.h"

#include "jblib/containers/vec.h"

MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  mag(::lattice->num_materials(), 4),
  s_transform_(globals::num_spins, 3),
  outfile(),
  m_stats_(),
  m2_stats_(),
  m4_stats_()
{
  using namespace globals;
  ::output->write("\ninitialising Magnetisation monitor\n");

  // create transform arrays for example to apply a Holstein Primakoff transform
  s_transform_.resize(num_spins, 3);

  libconfig::Setting& material_settings = ::config->lookup("materials");
  for (int i = 0; i < num_spins; ++i) {
    auto transform = jams::config_optional<Vec3>(material_settings[::lattice->atom_material(i)], "transform", jams::default_spin_transform);
    for (auto n = 0; n < 3; ++n) {
      s_transform_(i,n) = transform[n];
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

  for (int i = 0; i < lattice->num_materials(); ++i) {
    outfile << std::setw(12) << lattice->material_name(i) + ":mx" << "\t";
    outfile << std::setw(12) << lattice->material_name(i) + ":my" << "\t";
    outfile << std::setw(12) << lattice->material_name(i) + ":mz" << "\t";
    outfile << std::setw(12) << lattice->material_name(i) + ":m" << "\t";
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
      int type = lattice->atom_material(i);
      for (j = 0; j < 3; ++j) {
        mag(type, j) += s(i, j);
      }
    }

    for (i = 0; i < lattice->num_materials(); ++i) {
      for (j = 0; j < 3; ++j) {
        mag(i, j) = mag(i, j)/static_cast<double>(lattice->num_of_material(i));
      }
    }

    for (i = 0; i < lattice->num_materials(); ++i) {
      mag(i, 3) = sqrt(mag(i, 0)*mag(i, 0) + mag(i, 1)*mag(i, 1)
        + mag(i, 2)*mag(i, 2));
    }

    outfile << std::setw(12) << std::scientific << solver->time() << "\t";
    outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";

    for (i = 0; i < 3; ++i) {
      outfile <<  std::setw(12) << solver->physics()->applied_field(i) << "\t";
    }

    for (i = 0; i < lattice->num_materials(); ++i) {
      outfile << std::setw(12) << mag(i, 0) << "\t";
      outfile << std::setw(12) << mag(i, 1) << "\t";
      outfile << std::setw(12) << mag(i, 2) << "\t";
      outfile << std::setw(12) << mag(i, 3) << "\t";
    }

    if (convergence_is_on_ && solver->time() > convergence_burn_time_) {
      double m2 = binder_m2();
      m_stats_.add(sqrt(m2));
      m2_stats_.add(m2);
      m4_stats_.add(m2 * m2);


      outfile << std::setw(12) << m2 << "\t";
      outfile << std::setw(12) << m2 * m2 << "\t";
      outfile << std::setw(12) << binder_cumulant() << "\t";

      if (convergence_is_on_) {
        if (m_stats_.size() > 1 && m_stats_.size() % 10 == 0) {
          double diagnostic;
          m_stats_.geweke(diagnostic, convergence_stderr_);

          convergence_geweke_m_diagnostic_.push_back(diagnostic);

          outfile << std::setw(12) << diagnostic << "\t" << convergence_stderr_ << "\t" << m_stats_.stddev(0.1*m_stats_.size(), m_stats_.size()) / sqrt(m_stats_.size()*0.9);
        } else {
          outfile << std::setw(12) << "--------";
        }
      }
    }

    outfile << std::endl;
}

double MagnetisationMonitor::binder_m2() {
  using namespace globals;

  Vec3 m;

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      m[j] = m[j] + s_transform_(i, j) * s(i, j);
    }
  }

  return abs_sq(m) / square(static_cast<double>(num_spins));
}

double MagnetisationMonitor::binder_cumulant() {
  return 1.0 - (m4_stats_.mean()) / (3.0 * square(m2_stats_.mean()));
}

bool MagnetisationMonitor::is_converged() {

  if (convergence_geweke_m_diagnostic_.size() < 10) {
    return false;
  }

  int z_count = std::count_if(
    convergence_geweke_m_diagnostic_.begin() + 0.5 * convergence_geweke_m_diagnostic_.size(),
    convergence_geweke_m_diagnostic_.end(),
    [](double x) {return std::abs(x) < 1.96;});

  return (z_count > (0.95 * 0.5 * convergence_geweke_m_diagnostic_.size())  && convergence_stderr_ < convergence_tolerance_) && convergence_is_on_;
}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
