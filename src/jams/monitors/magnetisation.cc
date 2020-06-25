// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/interface/openmp.h"
#include "jams/helpers/output.h"

#include "jams/monitors/magnetisation.h"
#include "magnetisation.h"


MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  s_transform_(globals::num_spins),
  tsv_file(jams::output::full_path_filename("mag.tsv")),
  m_stats_(),
  m2_stats_(),
  m4_stats_()
{
  using namespace globals;

  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();

  s_transform_.resize(num_spins);
  for (int i = 0; i < num_spins; ++i) {
    s_transform_(i) = lattice->material(lattice->atom_material_id(i)).transform;
  }

  zero(material_count_.resize(lattice->num_materials()));
  for (auto i = 0; i < num_spins; ++i) {
    material_count_(lattice->atom_material_id(i))++;
  }
}

void MagnetisationMonitor::update(Solver * solver) {
  using namespace jams;
  using namespace globals;

  jams::MultiArray<Vec3, 1> magnetisation(::lattice->num_materials());
  zero(magnetisation);

  for (auto i = 0; i < num_spins; ++i) {
    const auto type = lattice->atom_material_id(i);
    for (auto j = 0; j < 3; ++j) {
      magnetisation(type)[j] += s(i, j);
    }
  }

  for (auto type = 0; type < lattice->num_materials(); ++type) {
    if (material_count_(type) == 0) continue;
    for (auto j = 0; j < 3; ++j) {
      magnetisation(type)[j] /= static_cast<double>(material_count_(type));
    }
  }

  tsv_file.width(12);
  tsv_file << fmt::sci << solver->time();
  tsv_file << fmt::decimal << solver->physics()->temperature();

  for (auto i = 0; i < 3; ++i) {
    tsv_file << fmt::decimal << solver->physics()->applied_field(i);
  }

  for (auto type = 0; type < lattice->num_materials(); ++type) {
    tsv_file << fmt::decimal << magnetisation(type);
    tsv_file << fmt::decimal << norm(magnetisation(type));
  }

  if (convergence_is_on_ && solver->time() > convergence_burn_time_) {
    double m2 = binder_m2();
    m_stats_.add(sqrt(m2));
    m2_stats_.add(m2);
    m4_stats_.add(m2 * m2);

    tsv_file << fmt::decimal << m2;
    tsv_file << fmt::decimal << m2 * m2;
    tsv_file << fmt::decimal << binder_cumulant();

    if (convergence_is_on_) {
      if (m_stats_.size() > 1 && m_stats_.size() % 10 == 0) {
        double diagnostic;
        m_stats_.geweke(diagnostic, convergence_stderr_);

        convergence_geweke_m_diagnostic_.push_back(diagnostic);

        tsv_file << fmt::decimal << diagnostic;
        tsv_file << fmt::decimal << convergence_stderr_;
        tsv_file << fmt::decimal << m_stats_.stddev(0.1*m_stats_.size(), m_stats_.size()) / sqrt(m_stats_.size()*0.9);
      } else {
        tsv_file << "--------";
      }
    }
  }
  tsv_file << std::endl;
}

double MagnetisationMonitor::binder_m2() {
  using namespace globals;

  Vec3 mag;

  for (int i = 0; i < num_spins; ++i) {
    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        mag[m] = mag[m] + s_transform_(i)[m][n] * s(i, n);
      }
    }
  }

  return norm_sq(mag) / square(static_cast<double>(num_spins));
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

std::string MagnetisationMonitor::tsv_header() {
  using namespace jams;

  std::stringstream ss;
  ss.width(12);

  ss << fmt::sci << "time";
  ss << fmt::decimal << "T";
  ss << fmt::decimal << "hx";
  ss << fmt::decimal << "hy";
  ss << fmt::decimal << "hz";

  for (auto i = 0; i < lattice->num_materials(); ++i) {
    auto name = lattice->material_name(i);
    for (const auto& suffix : {"_mx", "_my", "_mz", "_m"}) {
      ss << fmt::decimal << name + suffix;
    }
  }

  if (convergence_is_on_) {
    ss << fmt::decimal << "m2";
    ss << fmt::decimal << "m4";
    ss << fmt::decimal << "binder";
    ss << fmt::decimal << "geweke_m2";
    ss << fmt::decimal << "geweke_m4";
  }
  ss << std::endl;

  return ss.str();
}
