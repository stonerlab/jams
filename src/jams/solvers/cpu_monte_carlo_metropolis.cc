// Copyright 2014 Joseph Barker. All rights reserved.
#include <libconfig.h++>

#include "cpu_monte_carlo_metropolis.h"

#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/helpers/montecarlo.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/helpers/permutations.h"
#include "jams/helpers/output.h"
#include <jams/helpers/montecarlo.h>

#include <iomanip>

using namespace std;

void MetropolisMCSolver::initialize(const libconfig::Setting& settings) { //libconfig - A library for processing structured configuration files
  using namespace globals;

  // initialize base class
  Solver::initialize(settings);

  max_steps_ = jams::config_required<int>(settings, "max_steps"); //required to be passed by the config files , search and find the value named "max_steps"
  min_steps_ = jams::config_optional<int>(settings, "min_steps", jams::defaults::solver_min_steps); //jams is a namespace
  output_write_steps_ = jams::config_optional<int>(settings, "output_write_steps", output_write_steps_);

  cout << "    max_steps " << max_steps_ << "\n";
  cout << "    min_steps " << min_steps_ << "\n";

  move_names_.emplace_back("angle");
  const auto sigma = jams::config_optional<double>(settings, "move_angle_sigma", 0.5);
  move_weights_.push_back(jams::config_optional<double>(settings, "move_fraction_angle", 1.0));
  move_functions_.emplace_back(
      jams::montecarlo::MonteCarloAngleMove<jams::RandomGeneratorType>(&jams::instance().random_generator(), sigma));

  move_names_.emplace_back("uniform");
  move_weights_.push_back(jams::config_optional<double>(settings, "move_fraction_uniform", 0.0));
  move_functions_.emplace_back(
      jams::montecarlo::MonteCarloUniformMove<jams::RandomGeneratorType>(&jams::instance().random_generator()));

  move_names_.emplace_back("reflection");
  move_weights_.push_back(jams::config_optional<double>(settings, "move_fraction_reflection", 0.0));
  move_functions_.emplace_back(
      jams::montecarlo::MonteCarloReflectionMove());

  moves_accepted_.resize(move_functions_.size());
  moves_attempted_.resize(move_functions_.size());

}

void MetropolisMCSolver::run() {
  std::discrete_distribution<int> move_distribution(begin(move_weights_), end(move_weights_));

  auto move_index = move_distribution(jams::instance().random_generator());

  moves_attempted_[move_index] += globals::num_spins;
  moves_accepted_[move_index] += monte_carlo_step(move_functions_[move_index]);

  iteration_++;

  if (iteration_ % output_write_steps_ == 0) {
    output_move_statistics();

    // reset statistics
    fill(begin(moves_attempted_), end(moves_attempted_), 0);
    fill(begin(moves_accepted_), end(moves_accepted_), 0);
  }
}

int MetropolisMCSolver::monte_carlo_step(const MoveFunction& trial_spin_move) {
  int moves_accepted = 0;
  for (auto n = 0; n < globals::num_spins; ++n) {
    moves_accepted += metropolis_algorithm(trial_spin_move, jams::montecarlo::random_spin_index());
  }
  return moves_accepted;
}

int MetropolisMCSolver::metropolis_algorithm(const MoveFunction& trial_spin_move, const int spin_index) {
  const auto s_initial = jams::montecarlo::get_spin(spin_index);
  const auto s_final = trial_spin_move(s_initial);

  const auto deltaE = energy_difference(spin_index, s_initial, s_final);

  if (jams::montecarlo::accept_on_probability(deltaE, physics_module_->temperature())) {
    jams::montecarlo::set_spin(spin_index, s_final);
    return 1;
  }

  return 0;
}

double MetropolisMCSolver::energy_difference(const int spin_index,
                                             const Vec3 &initial_spin,
                                             const Vec3 &final_spin) {
  auto energy_difference = 0.0;
  for (const auto &ham : hamiltonians_) {
	  energy_difference += ham->calculate_energy_difference(spin_index, initial_spin, final_spin);
	}
	return energy_difference;
}

void MetropolisMCSolver::output_move_statistics() {
  if (!stats_file_.is_open()) {
    stats_file_.open(jams::output::full_path_filename("monte_carlo_stats.tsv"));
    stats_file_ << "iteration ";

    for (const auto name : move_names_) {
      stats_file_ << name << " ";
    }
    stats_file_ << endl;
  }

  stats_file_ << iteration() << " ";
  for (auto n = 0; n < move_functions_.size(); ++n) {
    stats_file_ << division_or_zero(moves_accepted_[n], moves_attempted_[n]) << " ";
  }
  stats_file_ << std::endl;
}
