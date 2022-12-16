// Copyright 2014 Joseph Barker. All rights reserved.

#include "cpu_llg_heun.h"

#include <cmath>
#include <jams/interface/config.h>
#include <jams/helpers/maths.h>
#include "jams/helpers/consts.h"

#include "jams/core/globals.h"
#include "jams/core/physics.h"
#include "jams/helpers/random.h"
#include "jams/interface/openmp.h"

using namespace std;

void HeunLLGSolver::initialize(const libconfig::Setting& settings) {
  // convert input in seconds to picoseconds for internal units
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  cout << "\ntimestep (ps) " << step_size_ << "\n";
  cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  s_old_.resize(globals::num_spins, 3);
  sigma_.resize(globals::num_spins);
  w_.resize(globals::num_spins, 3);

  bool use_gilbert_prefactor = jams::config_optional<bool>(globals::config->lookup("solver"), "gilbert_prefactor", false);
  cout << "    llg gilbert_prefactor " << use_gilbert_prefactor << "\n";

  for(int i = 0; i < globals::num_spins; ++i) {
    double denominator = 1.0;
    if (use_gilbert_prefactor) {
      denominator = 1.0 + pow2(globals::alpha(i));
    }
    sigma_(i) = sqrt((2.0 * kBoltzmannIU * globals::alpha(i)) /
                     (globals::mus(i) * globals::gyro(i) * globals::solver->time_step() * denominator));
  }
}

void HeunLLGSolver::run() {
  double t0 = time_;

  std::normal_distribution<> normal_distribution;

  // copy the spin configuration at the start of the step
  s_old_ = globals::s;

  if (physics_module_->temperature() > 0.0) {

    std::generate(w_.begin(), w_.end(), [&](){return normal_distribution(random_generator_);});

    const auto sqrt_temperature = sqrt(physics_module_->temperature());
    OMP_PARALLEL_FOR
    for (auto i = 0; i < globals::num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        w_(i, j) = w_(i, j) * sigma_(i) * sqrt_temperature;
      }
    }
  }

  Solver::compute_fields();

  if (physics_module_->temperature() > 0.0) {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < globals::num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        globals::h(i, j) = (w_(i, j) + globals::h(i, j) / globals::mus(i));
      }
    }
  } else {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < globals::num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        globals::h(i, j) = globals::h(i, j) / globals::mus(i);
      }
    }
  }

  OMP_PARALLEL_FOR
  for (auto i = 0; i < globals::num_spins; ++i) {
    Vec3 spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
    Vec3 field = {globals::h(i,0), globals::h(i,1), globals::h(i,2)};

    Vec3 rhs = -globals::gyro(i) * (cross(spin, field) + globals::alpha(i) * cross(spin, (cross(spin, field))));

    for (auto j = 0; j < 3; ++j) {
      globals::ds_dt(i, j) = 0.5 * rhs[j];
    }

    spin = unit_vector(spin + step_size_ * rhs);

     for (auto j = 0; j < 3; ++j) {
       globals::s(i, j) = spin[j];
    }

  }

  double mid_time_step = step_size_;
  time_ = t0 + mid_time_step;

  Solver::compute_fields();

  if (physics_module_->temperature() > 0.0) {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < globals::num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        globals::h(i, j) = (w_(i, j) + globals::h(i, j) / globals::mus(i));
      }
    }
  } else {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < globals::num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        globals::h(i, j) = globals::h(i, j) / globals::mus(i);
      }
    }
  }

  OMP_PARALLEL_FOR
  for (auto i = 0; i < globals::num_spins; ++i) {
    Vec3 spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
    Vec3 spin_old = {s_old_(i,0), s_old_(i,1), s_old_(i,2)};

    Vec3 field = {globals::h(i,0), globals::h(i,1), globals::h(i,2)};
    Vec3 rhs = -globals::gyro(i) * (cross(spin, field) + globals::alpha(i) * cross(spin, (cross(spin, field))));

    for (auto j = 0; j < 3; ++j) {
      globals::ds_dt(i, j) = globals::ds_dt(i, j) + 0.5 * rhs[j];
    }

    Vec3 ds = {globals::ds_dt(i, 0), globals::ds_dt(i, 1) , globals::ds_dt(i, 2)};

    spin = unit_vector(spin_old + step_size_ * ds);

    for (auto j = 0; j < 3; ++j) {
      globals::s(i, j) = spin[j];
    }

  }

  iteration_++;
  time_ = iteration_ * step_size_;
}
