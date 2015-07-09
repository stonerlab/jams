// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/metropolismc.h"

#include "core/consts.h"

#include "core/globals.h"

void MetropolisMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

    // initialize base class
  Solver::initialize(argc, argv, idt);

  output.write("Initialising Metropolis Monte-Carlo solver\n");

  jams_warning("Untested since re-write for version 1.0");

  output.write("  * Converting symmetric to general MAP matrices\n");

  // J1ij_t.convertSymmetric2General();

  // output.write("  * Converting MAP to CSR\n");

  // J1ij_t.convertMAP2CSR();

  // output.write("  * J1ij Tensor matrix memory (CSR): %f MB\n",
    // J1ij_t.calculateMemory());
}

void MetropolisMCSolver::oneSpinEnergy(const int &i, double total[3]) {
  using namespace globals;

  total[0] = 0.0; total[1] = 0.0; total[2] = 0.0;


  // exchange



  }

  void MetropolisMCSolver::run() {
    using namespace globals;

    const double theta = 0.1;
    const double Efactor = 0.671713067;  // muB/kB

    // pick spins randomly on average num_spins per step
    for (int n = 0; n < num_spins; ++n) {
      int i = rng.uniform()*(num_spins-1);

      double Enbr[3] = {0.0, 0.0, 0.0};
      oneSpinEnergy(i, Enbr);

      const double E1 = (Enbr[0]*s(i, 0) + Enbr[1]*s(i, 1) + Enbr[2]*s(i, 2));

        // trial move is random small angle
      double s_new[3];

      rng.sphere(s_new[0], s_new[1], s_new[2]);

      for (int j = 0; j < 3; ++j) {
        s_new[j] = s(i, j) + theta*s_new[j];
      }

        // normalise new spin
      const double norm =
        1.0/sqrt(s_new[0]*s_new[0] + s_new[1]*s_new[1] + s_new[2]*s_new[2]);
      for (int j = 0; j < 3; ++j) {
        s_new[j] = s_new[j]*norm;
      }

      const double E2 =
        (Enbr[0]*s_new[0] + Enbr[1]*s_new[1] + Enbr[2]*s_new[2]);

      double deltaE = E2-E1;

      if (deltaE < 0.0) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      } else if (rng.uniform() < exp(-(deltaE*Efactor)/physics_module_->temperature())) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      }
    }

    for (int n = 0; n < num_spins; ++n) {
      int i = rng.uniform()*(num_spins-1);

      double Enbr[3] = {0.0, 0.0, 0.0};
      oneSpinEnergy(i, Enbr);

      const double E1 = (Enbr[0]*s(i, 0) + Enbr[1]*s(i, 1) + Enbr[2]*s(i, 2));

      // trial move is random small angle
      const double s_new[3]={-s(i, 0), -s(i, 1), -s(i, 2)};

      const double E2 =
        (Enbr[0]*s_new[0] + Enbr[1]*s_new[1] + Enbr[2]*s_new[2]);
      double deltaE = E2-E1;

      if (deltaE < 0.0) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      } else if (rng.uniform() < exp(-(deltaE*Efactor)/physics_module_->temperature())) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      }
    }
    iteration_++;
  }

  void MetropolisMCSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s) {
  }
