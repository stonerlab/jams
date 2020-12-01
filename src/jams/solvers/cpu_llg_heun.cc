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
  using namespace globals;

  // initialize base class
  Solver::initialize(settings);

  time_step_ = jams::config_required<double>(settings, "t_step");
  dt = time_step_ * kGyromagneticRatio;

  auto t_max = jams::config_required<double>(settings, "t_max");
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0);

  max_steps_ = static_cast<int>(t_max / time_step_);
  min_steps_ = static_cast<int>(t_min / time_step_);

  cout << "\ntimestep " << dt << "\n";
  cout << "\nt_max " << t_max << " steps " << max_steps_ << "\n";
  cout << "\nt_min " << t_min << " steps " << min_steps_ << "\n";

  s_old_.resize(num_spins, 3);
  sigma_.resize(num_spins);
  w_.resize(num_spins, 3);

  bool use_gilbert_prefactor = jams::config_optional<bool>(config->lookup("solver"), "gilbert_prefactor", false);
  cout << "    llg gilbert_prefactor " << use_gilbert_prefactor << "\n";

  for(int i = 0; i < num_spins; ++i) {
    double denominator = 1.0;
    if (use_gilbert_prefactor) {
      denominator = 1.0 + pow2(globals::alpha(i));
    }
    sigma_(i) = sqrt((2.0 * kBoltzmann * globals::alpha(i) * globals::mus(i)) / (solver->time_step() * kGyromagneticRatio * kBohrMagneton * denominator) );
  }

  initialized_ = true;
}

void HeunLLGSolver::run() {
  using namespace globals;

  std::normal_distribution<> normal_distribution;

  // copy the spin configuration at the start of the step
  s_old_ = s;

  if (physics_module_->temperature() > 0.0) {

    std::generate(w_.begin(), w_.end(), [&](){return normal_distribution(random_generator_);});

    const auto sqrt_temperature = sqrt(physics_module_->temperature());
    OMP_PARALLEL_FOR
    for (auto i = 0; i < num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        w_(i, j) = w_(i, j) * sigma_(i) * sqrt_temperature;
      }
    }
  }

  Solver::compute_fields();

  if (physics_module_->temperature() > 0.0) {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        h(i, j) = (w_(i, j) + h(i, j) + (physics_module_->applied_field(j)) * mus(i)) * gyro(i);
      }
    }
  } else {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        h(i, j) = (h(i, j) + (physics_module_->applied_field(j))*mus(i))*gyro(i);
      }
    }
  }

  OMP_PARALLEL_FOR
  for (auto i = 0; i < num_spins; ++i) {
    Vec3 spin = {s(i,0), s(i,1), s(i,2)};
    Vec3 field = {h(i,0), h(i,1), h(i,2)};

    Vec3 rhs = cross(spin, field) + alpha(i) * cross(spin, (cross(spin, field)));

    for (auto j = 0; j < 3; ++j) {
       ds_dt(i, j) = 0.5 * rhs[j];
    }

    spin = unit_vector(spin + dt * rhs);

     for (auto j = 0; j < 3; ++j) {
      s(i, j) = spin[j];
    }

  }

  Solver::compute_fields();

  if (physics_module_->temperature() > 0.0) {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        h(i, j) = (w_(i, j) + h(i, j) + (physics_module_->applied_field(j)) * mus(i)) * gyro(i);
      }
    }
  } else {
    OMP_PARALLEL_FOR
    for (auto i = 0; i < num_spins; ++i) {
      for (auto j = 0; j < 3; ++j) {
        h(i, j) = (h(i, j) + (physics_module_->applied_field(j)) * mus(i)) * gyro(i);
      }
    }
  }

  OMP_PARALLEL_FOR
  for (auto i = 0; i < num_spins; ++i) {
    Vec3 spin = {s(i,0), s(i,1), s(i,2)};
    Vec3 spin_old = {s_old_(i,0), s_old_(i,1), s_old_(i,2)};

    Vec3 field = {h(i,0), h(i,1), h(i,2)};
    Vec3 rhs = cross(spin, field) + alpha(i) * cross(spin, (cross(spin, field)));

    for (auto j = 0; j < 3; ++j) {
      ds_dt(i, j) = ds_dt(i, j) + 0.5 * rhs[j];
    }

    Vec3 ds = {ds_dt(i, 0), ds_dt(i, 1) , ds_dt(i, 2)};

    spin = unit_vector(spin_old + dt * ds);

    for (auto j = 0; j < 3; ++j) {
      s(i, j) = spin[j];
    }

  }

  iteration_++;
}
