// Copyright 2014 Joseph Barker. All rights reserved.

#include <jams/solvers/cpu_monte_carlo_metropolis.h>

#include <iomanip>
#include <algorithm>

#include <libconfig.h++>

#include <jams/core/globals.h>
#include <jams/core/hamiltonian.h>
#include <jams/core/physics.h>
#include <jams/core/solver.h>
#include <jams/helpers/montecarlo.h>
#include <jams/helpers/output.h>
#include <jams/interface/config.h>

using namespace std;

void MetropolisMCSolver::initialize(const libconfig::Setting& settings) {
  using namespace globals;

  max_steps_ = jams::config_required<int>(settings, "max_steps");
  min_steps_ = jams::config_optional<int>(settings, "min_steps", jams::defaults::solver_min_steps);
  output_write_steps_ = jams::config_optional<int>(settings, "output_write_steps", output_write_steps_);


  use_random_spin_order_ = jams::config_optional<bool>(settings, "use_random_spin_order", true);
  cout << "    use_random_spin_order " << std::boolalpha << use_random_spin_order_ << "\n";

  use_total_energy_ = jams::config_optional<bool>(settings, "use_total_energy", false);
  cout << "    use_total_energy " << std::boolalpha << use_total_energy_ << "\n";

  cout << "    max_steps " << max_steps_ << "\n";
  cout << "    min_steps " << min_steps_ << "\n";

  if (settings.exists("move_fraction_uniform") || settings.exists("move_fraction_angle") || settings.exists("move_fraction_reflection")) {
    move_fraction_uniform_    = jams::config_optional<double>(settings, "move_fraction_uniform", 0.0);
    move_fraction_angle_      = jams::config_optional<double>(settings, "move_fraction_angle", 0.0);
    move_fraction_reflection_ = jams::config_optional<double>(settings, "move_fraction_reflection", 0.0);
    move_angle_sigma_         = jams::config_optional<double>(settings, "move_angle_sigma", 0.5);

    double move_fraction_sum = move_fraction_uniform_ + move_fraction_angle_ + move_fraction_reflection_;

  // Create a set of vectors which contain different types of Monte Carlo moves.
  // Each move can has a 'fraction' (weight) associated with it to allow some
  // move types to be attempted more frequently than others.
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
  // Randomly choose an index for the move type. This is not a uniform
  // distribution but will give discrete integers based on the move_weights_.
  std::discrete_distribution<int> move_distribution(begin(move_weights_), end(move_weights_));
  auto move_index = move_distribution(jams::instance().random_generator());

  // Perform a Monte Carlo step with the chosen move and record some statistics.
  // NOTE: every trial move for the step uses the same randomly selected move
  // function.
  moves_attempted_[move_index] += globals::num_spins;
  moves_accepted_[move_index] += monte_carlo_step(move_functions_[move_index]);

  iteration_++;

  // Output statistics to file at the configured interval
  if (iteration_ % output_write_steps_ == 0) {
    output_move_statistics();

    // Reset statistics
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

  if (jams::montecarlo::accept_on_boltzmann_distribution(deltaE,
                                                         physics_module_->temperature())) {
    accept_move(spin_index, s_initial, s_final);
    return 1;
  }

  return 0;
}

double MetropolisMCSolver::energy_difference(const int spin_index,
                                             const Vec3 &initial_spin,
                                             const Vec3 &final_spin) {
  auto energy_difference = 0.0;
  // Calculate the energy difference from all of the Hamiltonian terms
  for (const auto &ham : hamiltonians_) {
	  energy_difference += ham->calculate_energy_difference(spin_index, initial_spin, final_spin);
	}
	return energy_difference;
}


void MetropolisMCSolver::output_move_statistics() {
  if (!stats_file_.is_open()) {
    stats_file_.open(jams::output::full_path_filename("monte_carlo_stats.tsv"));
    stats_file_ << "iteration ";

    for (const auto& name : move_names_) {
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

void
MetropolisMCSolver::accept_move(const int spin_index, const Vec3 &initial_spin,
                                const Vec3 &final_spin) {
  // The trial move has been accepted so set the spin to the new value
  jams::montecarlo::set_spin(spin_index, final_spin);
}
