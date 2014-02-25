// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/heunllg.h"

#include <cmath>

#include "core/consts.h"
#include "core/fields.h"
#include "core/globals.h"

#ifdef MKL
#include <mkl_spblas.h>
#endif


void HeunLLGSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

  // initialize base class
  Solver::initialize(argc, argv, idt);

  ::output.write("Initialising Heun LLG solver (CPU)\n");

  snew.resize(num_spins, 3);
  sigma.resize(num_spins);
  eng.resize(num_spins, 3);
  w.resize(num_spins, 3);

  for (int i = 0; i < num_spins; ++i) {
    sigma(i) = sqrt((2.0*boltzmann_si*alpha(i))/(time_step_*mus(i)*mu_bohr_si));
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
        w(i, j) = (rng.normal())*sigma(i)*stmp*mus(i)*gyro(i); // MOVE THESE INTO SIGMA
      }
    }
  }


  Solver::compute_fields();

  if (physics_module_->temperature() > 0.0) {
    for (i = 0; i < num_spins; ++i) {
      for (j = 0; j < 3; ++j) {
        h(i, j) += w(i,j);
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
      snew(i, j) = s(i, j) + 0.5*time_step_*rhs[j];
    }

    for (j = 0; j < 3; ++j) {
      s(i, j) = s(i, j) + time_step_*rhs[j];
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
        h(i, j) += w(i,j);
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
      s(i, j) = snew(i, j) + 0.5*time_step_*rhs[j];
    }

    norm = 1.0/sqrt(s(i, 0)*s(i, 0) + s(i, 1)*s(i, 1) + s(i, 2)*s(i, 2));

    for (j = 0; j < 3; ++j) {
      s(i, j) = s(i, j)*norm;
    }
  }

  iteration_++;
}

void HeunLLGSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s) {
  using namespace globals;

  e1_s = 0.0; e1_t = 0.0; e2_s = 0.0; e2_t = 0.0;

  if (J1ij_s.nonZero() > 0) {
    std::fill(eng.data(), eng.data()+num_spins3, 0.0);
    compute_bilinear_scalar_interactions_csr(J1ij_s.valPtr(), J1ij_s.colPtr(), J1ij_s.ptrB(),
      J1ij_s.ptrE(), eng);
    for (int i = 0; i < num_spins; ++i) {
      e1_s = e1_s + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }
    e1_s = e1_s/num_spins;
  }
  if (J1ij_t.nonZero() > 0) {
    std::fill(eng.data(), eng.data()+num_spins3, 0.0);
    compute_bilinear_tensor_interactions_csr(J1ij_t.valPtr(), J1ij_t.colPtr(), J1ij_t.ptrB(),
      J1ij_t.ptrE(), eng);
    for (int i = 0; i < num_spins; ++i) {
      e1_t = e1_t + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }
    e1_t = e1_t/num_spins;
  }
  if (J2ij_s.nonZero() > 0) {
    std::fill(eng.data(), eng.data()+num_spins3, 0.0);
    compute_biquadratic_scalar_interactions_csr(J2ij_s.valPtr(), J2ij_s.colPtr(), J2ij_s.ptrB(),
      J2ij_s.ptrE(), eng);
    for (int i = 0; i < num_spins; ++i) {
      e2_s = e2_s + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }

    e2_s = 0.5*e2_s/num_spins;
  }
  if (J2ij_t.nonZero() > 0) {
    std::fill(eng.data(), eng.data()+num_spins3, 0.0);
    compute_biquadratic_tensor_interactions_csr(J2ij_t.valPtr(), J2ij_t.colPtr(), J2ij_t.ptrB(),
      J2ij_t.ptrE(), eng);
    for (int i = 0; i < num_spins; ++i) {
      e2_t = e2_t + (s(i, 0)*eng(i, 0)+s(i, 1)*eng(i, 1)+s(i, 2)*eng(i, 2));
    }

    e2_t = 0.5*e2_t/num_spins;
  }
}
