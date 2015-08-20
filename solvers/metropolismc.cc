// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/metropolismc.h"

#include "core/consts.h"
#include "core/globals.h"
#include "core/montecarlo.h"
#include "core/hamiltonian.h"

void MetropolisMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

  // initialize base class
  Solver::initialize(argc, argv, idt);

  output.write("Initialising Metropolis Monte-Carlo solver\n");
}

  void MetropolisMCSolver::run() {
    using namespace globals;

    // 0 UTS
    // 1 STS
    // 2 UTS
    // 3 RTS
    // 4 UTS
    // 5 STS
    // 6 UTS
    // 7 RTS
    // 8 UTS
    // 9 STS

    if (iteration_ % 2 == 0) {
      MetropolisAlgorithm(mc_uniform_trial_step);
    } else {
      if ((iteration_ - 1) % 4 == 0) {
        MetropolisAlgorithm(mc_reflection_trial_step);
      } else {
        MetropolisAlgorithm(mc_small_trial_step);
      }
    }
    iteration_++;
  }

  void MetropolisMCSolver::MetropolisAlgorithm(jblib::Vec3<double> (*mc_trial_step)(const jblib::Vec3<double>)) {
    const double beta = kBohrMagneton/(kBoltzmann*physics_module_->temperature());
    int n, random_spin_number;
    double deltaE = 0.0;
    jblib::Vec3<double> s_initial, s_final;

    for (n = 0; n < globals::num_spins; ++n) {
      random_spin_number = rng.uniform_discrete(0, globals::num_spins - 1);
      s_initial = mc_spin_as_vec(random_spin_number);
      s_final = mc_trial_step(s_initial);

      deltaE = 0.0;
      for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
        deltaE += (*it)->calculate_one_spin_energy_difference(random_spin_number, s_initial, s_final);
      }

      if (deltaE < 0.0) {
        mc_set_spin_as_vec(random_spin_number, s_final);
        continue;
      }

      if (rng.uniform() < exp(-deltaE*beta)) {
        mc_set_spin_as_vec(random_spin_number, s_final);
      }
    }
  }