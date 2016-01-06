// Copyright 2015 Joseph Barker. All rights reserved.

#include "core/globals.h"

#include "core/consts.h"
#include "core/montecarlo.h"
#include "core/hamiltonian.h"

#include "solvers/monte-carlo-wolff.h"

MonteCarloWolffSolver::MonteCarloWolffSolver()
  : attempted_moves_(0),
    cluster_size_(0),
    neighbours_(globals::num_spins)
{
}

void MonteCarloWolffSolver::initialize(int argc, char **argv, double idt) {
  // initialize base class
  Solver::initialize(argc, argv, idt);

  r_cutoff_ = 1.0;

  if (config.exists("sim.r_cutoff")) {
      r_cutoff_ = config.lookup("sim.r_cutoff");
  }

  neighbours_.resize(globals::num_spins);

  std::vector<Atom> nbr;
  for (int i = 0; i < globals::num_spins; ++i) {
    lattice.atom_neighbours(i, r_cutoff_, nbr);
    for (auto it = nbr.begin(); it != nbr.end(); ++it) {
      neighbours_[i].push_back((*it).id);
    }
  }

}

void MonteCarloWolffSolver::run() {

  if (iteration_ % 4 == 0) {
  int size = 0;
  attempted_moves_ = 0;
  while(attempted_moves_ < globals::num_spins) {
    size = cluster_move();
  }
} else {

  const double beta = kBohrMagneton/(kBoltzmann*physics_module_->temperature());
  int n, random_spin_number;
  double deltaE = 0.0;
  jblib::Vec3<double> s_initial, s_final;

  for (n = 0; n < globals::num_spins; ++n) {

    // 2015-12-10 (JB) striding uniformly is ~4x faster than random choice (clang OSX).
    // Seems to be because of caching/predication in the exchange field calculation.
    random_spin_number = n; //rng.uniform_discrete(0, globals::num_spins - 1);

    s_initial = mc_spin_as_vec(random_spin_number);
    s_final = mc_uniform_trial_step(s_initial);

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

  iteration_++;
}

Vec3 wolff_reflection(Vec3 s, const Vec3 &r) {
  s = s - r * 2.0 * dot(s, r);
  return s;
}

void reset_cluster(const std::vector<int>& cluster_map, const Vec3& r) {
  for (auto it = cluster_map.begin(); it != cluster_map.end(); ++it) {
    mc_set_spin_as_vec(*it, wolff_reflection(mc_spin_as_vec(*it), -r));
  }
}

int MonteCarloWolffSolver::cluster_move() {
  const unsigned int potential_energy_size_check = 10;
  const double       beta = kBohrMagneton / (kBoltzmann * physics_module_->temperature());

  int x, y;
  Vec3 r_rand, spin_old, spin_trial;

  std::vector<int>  cluster_map;
  std::queue<int>   cluster_processing_queue;

  std::vector<bool> is_attached(globals::num_spins, false);

  double potential_energy = 0.0;
  double bond_energy      = 0.0;

  // choose a random vector for the reflection plane normal
  rng.sphere(r_rand[0], r_rand[1], r_rand[2]);

  // choose an initial site at random
  x = rng.uniform_discrete(0, globals::num_spins - 1);

  is_attached[x] = true;
  cluster_map.push_back(x);
  cluster_processing_queue.push(x);

  spin_old   = mc_spin_as_vec(x);
  spin_trial = wolff_reflection(spin_old, r_rand);
  mc_set_spin_as_vec(x, spin_trial);

  for (auto it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    potential_energy += (*it)->calculate_bond_energy_difference(x, x, spin_old, spin_trial);
  }

  // processes the cluster
  while (!cluster_processing_queue.empty()) {

    x = cluster_processing_queue.front();
    cluster_processing_queue.pop();  // remove x from queue

    // for y in x neighbours
    for (auto nbr = neighbours_[x].begin(); nbr != neighbours_[x].end(); ++nbr) {
      y = *nbr;

      if (is_attached[y]) {
        continue;
      }

      spin_old   = mc_spin_as_vec(y);
      spin_trial = wolff_reflection(spin_old, r_rand);

      bond_energy = 0.0;
      for (auto it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
        bond_energy += (*it)->calculate_bond_energy_difference(x, y, spin_old, spin_trial);
      }

      if (rng.uniform() < mc_percolation_probability(bond_energy, beta)) {

        for (auto it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
          potential_energy += (*it)->calculate_bond_energy_difference(y, y, spin_old, spin_trial);
        }

        // add y to cluster
        is_attached[y] = true;
        cluster_map.push_back(y);
        mc_set_spin_as_vec(y, spin_trial);

        if (cluster_map.size() % potential_energy_size_check == 0
          && rng.uniform() > mc_boltzmann_probability(potential_energy, beta)) {
          reset_cluster(cluster_map, r_rand);
          return 0;
        }

        // add y to the processing list
        cluster_processing_queue.push(y);
      }

      attempted_moves_++;
    } // for neighbours
  }  // while cluster queue not empty

  // finished growing cluster

  if (rng.uniform() > mc_boltzmann_probability(potential_energy, beta)) {
    reset_cluster(cluster_map, r_rand);
    return 0;
  }

  return cluster_map.size();
}

MonteCarloWolffSolver::~MonteCarloWolffSolver() {

}