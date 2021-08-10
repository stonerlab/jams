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

#include <iomanip>

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

    move_fraction_uniform_     /= move_fraction_sum;
    move_fraction_angle_       /= move_fraction_sum;
    move_fraction_reflection_  /= move_fraction_sum;
  }
}

  void MetropolisMCSolver::run() {
    using namespace globals;
    std::uniform_real_distribution<> uniform_distribution;

    MonteCarloUniformMove<jams::RandomGeneratorType> uniform_move(&jams::instance().random_generator());
    MonteCarloAngleMove<jams::RandomGeneratorType>   angle_move(&jams::instance().random_generator(), move_angle_sigma_);
    MonteCarloReflectionMove           reflection_move;

    const double uniform_random_number = uniform_distribution(jams::instance().random_generator());
    if (uniform_random_number < move_fraction_uniform_) {
      if (use_total_energy_) {
        move_running_acceptance_count_uniform_ += MetropolisAlgorithmTotalEnergy(uniform_move);
      } else {
        move_running_acceptance_count_uniform_ += MetropolisAlgorithm(uniform_move);
      }
      run_count_uniform_++;
    } else if (uniform_random_number < (move_fraction_uniform_ + move_fraction_angle_)) {
      if (use_total_energy_) {
        move_running_acceptance_count_angle_ += MetropolisAlgorithmTotalEnergy(angle_move);
      } else {
        move_running_acceptance_count_angle_ += MetropolisAlgorithm(angle_move);
      }
      run_count_angle_++;
    } else {
      if (use_total_energy_) {
        move_running_acceptance_count_reflection_ += MetropolisAlgorithmTotalEnergy(reflection_move);
      } else {
        move_running_acceptance_count_reflection_ += MetropolisAlgorithm(reflection_move);
      }
      run_count_reflection_++;
    }

    iteration_++;

    if (iteration_ % output_write_steps_ == 0) {

      move_total_count_uniform_ += run_count_uniform_;
      move_total_count_angle_ += run_count_angle_;
      move_total_count_reflection_ += run_count_reflection_;

      move_total_acceptance_count_uniform_ += move_running_acceptance_count_uniform_;
      move_total_acceptance_count_angle_ += move_running_acceptance_count_angle_;
      move_total_acceptance_count_reflection_ += move_running_acceptance_count_reflection_;

      cout << "\n";
      cout << "iteration" << iteration_ << "\n";
      cout << "move_acceptance_fraction\n";

      cout << "  uniform ";
      cout << division_or_zero(move_running_acceptance_count_uniform_, globals::num_spins * run_count_uniform_) << " (";
      cout << division_or_zero(move_total_acceptance_count_uniform_, globals::num_spins * move_total_count_uniform_)
           << ") \n";

      cout << "  angle ";
      cout << division_or_zero(move_running_acceptance_count_angle_, globals::num_spins * run_count_angle_) << " (";
      cout << division_or_zero(move_total_acceptance_count_angle_, globals::num_spins * move_total_count_angle_)
           << ") \n";

      cout << "  reflection ";
      cout << division_or_zero(move_running_acceptance_count_reflection_, globals::num_spins * run_count_reflection_)
           << " (";
      cout << division_or_zero(move_total_acceptance_count_reflection_,
              globals::num_spins * move_total_count_reflection_) << ") \n";

      move_running_acceptance_count_uniform_ = 0;
      move_running_acceptance_count_angle_ = 0;
      move_running_acceptance_count_reflection_ = 0;

      run_count_uniform_ = 0;
      run_count_angle_ = 0;
      run_count_reflection_ = 0;
    }
  }

  void MetropolisMCSolver::MetropolisPreconditioner(std::function<Vec3(Vec3)>  trial_spin_move) {
    int n;
    double e_initial, e_final;
    Vec3 s_initial, s_final;

    s_initial = mc_spin_as_vec(0);
    s_final = trial_spin_move(s_initial);

    e_initial = 0.0;
    for (const auto& hamiltonian : hamiltonians_) {
      e_initial += hamiltonian->calculate_total_energy();
    }

    for (n = 0; n < globals::num_spins; ++n) {
      mc_set_spin_as_vec(n, s_final);
    }

    e_final = 0.0;
    for (const auto& hamiltonian : hamiltonians_) {
      e_final += hamiltonian->calculate_total_energy();
    }

    if (e_final - e_initial > 0.0) {
      for (n = 0; n < globals::num_spins; ++n) {
        mc_set_spin_as_vec(n, s_initial);
      }
    }
  }

  int MetropolisMCSolver::MetropolisAlgorithm(std::function<Vec3(Vec3)> trial_spin_move) {
    using std::min;
    using std::exp;
    std::uniform_real_distribution<> uniform_distribution;

    const double beta = 1.0 / (kBoltzmannIU * physics_module_->temperature());

    unsigned moves_accepted = 0;
    for (auto n = 0; n < globals::num_spins; ++n) {
      // 2015-12-10 (JB) striding uniformly is ~4x faster than random choice (clang OSX).
      // Seems to be because of caching/predication in the exchange field calculation.
      int spin_index = n;

      if (use_random_spin_order_) {
        spin_index = jams::instance().random_generator()(globals::num_spins);
      }

      // vacancy sites have spin components zero and mus zero and must be skipped
      // to avoid creating a moment on these sites with the trial move
      if (globals::mus(spin_index) == 0.0) {
        return 0.0;
      }

      auto s_initial = mc_spin_as_vec(spin_index);
      auto s_final = trial_spin_move(s_initial);

      auto deltaE = 0.0;
      for (const auto& ham : hamiltonians_) {
        deltaE += ham->calculate_energy_difference(spin_index, s_initial, s_final);
      }

      if (uniform_distribution(jams::instance().random_generator()) < exp(min(0.0, -deltaE * beta))) {
        mc_set_spin_as_vec(spin_index, s_final);
        moves_accepted++;
        continue;
      }
    }
    return moves_accepted;
  }

int MetropolisMCSolver::MetropolisAlgorithmTotalEnergy(std::function<Vec3(Vec3)> trial_spin_move) {
  using std::min;
  using std::exp;
  std::uniform_real_distribution<> uniform_distribution;

  const double beta = 1.0 / (kBoltzmannIU * physics_module_->temperature());

  int moves_accepted = 0;
  for (auto n = 0; n < globals::num_spins; ++n) {
    // 2015-12-10 (JB) striding uniformly is ~4x faster than random choice (clang OSX).
    // Seems to be because of caching/predication in the exchange field calculation.
    int spin_index = n;

    if (use_random_spin_order_) {
      spin_index = jams::instance().random_generator()(globals::num_spins);
    }

    auto s_initial = mc_spin_as_vec(spin_index);
    auto s_final = trial_spin_move(s_initial);

    auto e_initial = 0.0;
    for (const auto& ham : hamiltonians_) {
      e_initial += ham->calculate_total_energy();
    }

    mc_set_spin_as_vec(spin_index, s_final);
    auto e_final = 0.0;
    for (const auto& ham : hamiltonians_) {
      e_final += ham->calculate_total_energy();
    }

    auto deltaE = e_final - e_initial;

    if (uniform_distribution(jams::instance().random_generator()) < exp(min(0.0, -deltaE * beta))) {
      moves_accepted++;
      continue;
    }

    mc_set_spin_as_vec(spin_index, s_initial);
  }
  return moves_accepted;
}
