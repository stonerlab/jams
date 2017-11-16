// Copyright 2014 Joseph Barker. All rights reserved.

#include "cpu_llg_heun.h"

#include <cmath>
#include <jams/interface/config.h>

#include "jams/helpers/consts.h"

#include "jams/core/globals.h"
#include "jams/core/output.h"
#include "jams/core/physics.h"
#include "jams/core/rand.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


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

  output->write("\ntimestep\n  %1.8e\n", dt);
  output->write("\nt_max\n  %1.8e (%lu steps)\n", t_max, max_steps_);
  output->write("\nt_min\n  %1.8e (%lu steps)\n", t_min, min_steps_);

  snew.resize(num_spins, 3);
  sigma.resize(num_spins);
  eng.resize(num_spins, 3);
  w.resize(num_spins, 3);

  for (int i = 0; i < num_spins; ++i) {
    sigma(i) = sqrt( (2.0 * kBoltzmann * globals::alpha(i) * globals::mus(i)) / (solver->time_step() * kBohrMagneton) );
  }

  initialized_ = true;
}

void HeunLLGSolver::run() {
  using namespace globals;

  int i, j;
  double sxh[3], rhs[3];
  double norm;

  if (physics_module_->temperature() > 0.0) {
    const double stmp = sqrt(physics_module_->temperature());
    for (i = 0; i < num_spins; ++i) {
      for (j = 0; j < 3; ++j) {
        w(i, j) = (rng->normal())*sigma(i) * stmp; // MOVE THESE INTO SIGMA
      }
    }
  }


  Solver::compute_fields();

  if (physics_module_->temperature() > 0.0) {
    for (i = 0; i < num_spins; ++i) {
      for (j = 0; j < 3; ++j) {
        h(i, j) = (w(i,j) + h(i, j) + (physics_module_->applied_field(j))*mus(i))*gyro(i);
      }
    }
  } else {
    for (i = 0; i < num_spins; ++i) {
      for (j = 0; j < 3; ++j) {
        h(i, j) = (h(i, j) + (physics_module_->applied_field(j))*mus(i))*gyro(i);
      }
    }
  }

  for (i = 0; i < num_spins; ++i) {
    sxh[0] = s(i, 1)*h(i, 2) - s(i, 2)*h(i, 1);
    sxh[1] = s(i, 2)*h(i, 0) - s(i, 0)*h(i, 2);
    sxh[2] = s(i, 0)*h(i, 1) - s(i, 1)*h(i, 0);

    rhs[0] = sxh[0] + alpha(i) * (s(i, 1)*sxh[2] - s(i, 2)*sxh[1]);
    rhs[1] = sxh[1] + alpha(i) * (s(i, 2)*sxh[0] - s(i, 0)*sxh[2]);
    rhs[2] = sxh[2] + alpha(i) * (s(i, 0)*sxh[1] - s(i, 1)*sxh[0]);

    for (j = 0; j < 3; ++j) {
      snew(i, j) = s(i, j) + 0.5*dt*rhs[j];
    }

    for (j = 0; j < 3; ++j) {
      s(i, j) = s(i, j) + dt*rhs[j];
    }

    norm = 1.0/sqrt(s(i, 0)*s(i, 0) + s(i, 1)*s(i, 1) + s(i, 2)*s(i, 2));

    for (j = 0; j < 3; ++j) {
      s(i, j) = s(i, j)*norm;
    }
  }

  Solver::compute_fields();

  if (physics_module_->temperature() > 0.0) {
    for (i = 0; i < num_spins; ++i) {
      for (j = 0; j < 3; ++j) {
        h(i, j) = (w(i,j) + h(i, j) + (physics_module_->applied_field(j))*mus(i))*gyro(i);
      }
    }
  } else {
    for (i = 0; i < num_spins; ++i) {
      for (j = 0; j < 3; ++j) {
        h(i, j) = (h(i, j) + (physics_module_->applied_field(j))*mus(i))*gyro(i);
      }
    }
  }

  for (i = 0; i < num_spins; ++i) {
    sxh[0] = s(i, 1)*h(i, 2) - s(i, 2)*h(i, 1);
    sxh[1] = s(i, 2)*h(i, 0) - s(i, 0)*h(i, 2);
    sxh[2] = s(i, 0)*h(i, 1) - s(i, 1)*h(i, 0);

    rhs[0] = sxh[0] + alpha(i) * (s(i, 1)*sxh[2] - s(i, 2)*sxh[1]);
    rhs[1] = sxh[1] + alpha(i) * (s(i, 2)*sxh[0] - s(i, 0)*sxh[2]);
    rhs[2] = sxh[2] + alpha(i) * (s(i, 0)*sxh[1] - s(i, 1)*sxh[0]);

    for (j = 0; j < 3; ++j) {
      s(i, j) = snew(i, j) + 0.5*dt*rhs[j];
    }

    norm = 1.0/sqrt(s(i, 0)*s(i, 0) + s(i, 1)*s(i, 1) + s(i, 2)*s(i, 2));

    for (j = 0; j < 3; ++j) {
      s(i, j) = s(i, j)*norm;
    }
  }

  iteration_++;
}
