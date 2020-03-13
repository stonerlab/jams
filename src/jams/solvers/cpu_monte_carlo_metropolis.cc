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

  // initialize base class
  Solver::initialize(settings);

  max_steps_ = jams::config_required<int>(settings, "max_steps");
  min_steps_ = jams::config_optional<int>(settings, "min_steps", jams::defaults::solver_min_steps);
  output_write_steps_ = jams::config_optional<int>(settings, "output_write_steps", output_write_steps_);

  use_random_spin_order_ = jams::config_optional<bool>(settings, "use_random_spin_order", true);
  cout << "    use_random_spin_order " << std::boolalpha << use_random_spin_order_ << "\n";

  use_total_energy_ = jams::config_optional<bool>(settings, "use_total_energy", false);
  cout << "    use_total_energy " << std::boolalpha << use_total_energy_ << "\n";

  is_preconditioner_enabled_ = settings.exists("preconditioner_theta") || settings.exists("preconditioner_phi");
  preconditioner_delta_theta_ = jams::config_optional<double>(settings, "preconditioner_theta", 5.0);
  preconditioner_delta_phi_ = jams::config_optional<double>(settings, "preconditioner_phi", 5.0);

  cout << "    max_steps " << max_steps_ << "\n";
  cout << "    min_steps " << min_steps_ << "\n";
  cout << "    preconditioner " << is_preconditioner_enabled_ << "\n";

  if (is_preconditioner_enabled_) {
    cout << "    preconditioner_theta " << preconditioner_delta_theta_ << "\n";
    cout << "    preconditioner_phi   " << preconditioner_delta_phi_ << "\n";
  }

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

    if (is_preconditioner_enabled_ && iteration_ == 0) {
      cout << "preconditioning\n";

      cout << "  thermalizing\n";
      // do a short thermalization
      if (use_total_energy_) {
        for (int i = 0; i < 500; ++i) {
          MetropolisAlgorithmTotalEnergy(uniform_move);
        }
      } else {
        for (int i = 0; i < 500; ++i) {
          MetropolisAlgorithm(uniform_move);
        }
      }

      // now try systematic rotations
      cout << "  magnetization rotations\n";
      SystematicPreconditioner(preconditioner_delta_theta_, preconditioner_delta_phi_);
      cout << "done\n";
    }

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
    for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
      e_initial += (*it)->calculate_total_energy();
    }

    for (n = 0; n < globals::num_spins; ++n) {
      mc_set_spin_as_vec(n, s_final);
    }

    e_final = 0.0;
    for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
      e_final += (*it)->calculate_total_energy();
    }

    if (e_final - e_initial > 0.0) {
      for (n = 0; n < globals::num_spins; ++n) {
        mc_set_spin_as_vec(n, s_initial);
      }
    }
  }

  class MetropolisMCSolver::MagnetizationRotationMinimizer
  {
      std::vector<Hamiltonian*> * hamiltonians_;

      std::uint64_t count;
      double e_min;
      jams::MultiArray<double,2> s_min;
  public:
      explicit MagnetizationRotationMinimizer(std::vector<Hamiltonian*> & hamiltonians_ ) :
        hamiltonians_(&hamiltonians_), count(0), e_min(1e10), s_min(globals::s) {}

      jams::MultiArray<double,2> s() {
        return s_min;
      }

      template <class It>
          bool operator()(It first, It last)  // called for each permutation
          {
            using std::vector;

            int i, j;
            double energy;
            Vec3 s_new;
            vector<Mat3> rotation(::lattice->num_materials());
            vector<Vec3> mag(::lattice->num_materials());

            if (last - first != ::lattice->num_materials()) {
              throw std::runtime_error("number of angles in preconditioner does not match the number of materials");
            }

            // count the number of times this is called
            ++count;

            // calculate magnetization vector of each material
            for (i = 0; i < globals::num_spins; ++i) {
              for (j = 0; j < 3; ++j) {
                mag[::lattice->atom_material_id(i)][j] += globals::s(i, j);
              }
            }
            // don't need to normalize magnetization because only the direction is important
            // calculate rotation matrix between magnetization and desired direction
            for (i = 0; i < ::lattice->num_materials(); ++i) {
              rotation[i] = rotation_matrix_between_vectors(mag[i], spherical_to_cartesian_vector(1.0, *first, 0.0));
              ++first;
            }

            for (i = 0; i < globals::num_spins; ++i) {
              for (j = 0; j < 3; ++j) {
                s_new[j] = globals::s(i, j);
              }
              s_new = rotation[::lattice->atom_material_id(i)] * s_new;
              for (j = 0; j < 3; ++j) {
                globals::s(i, j) = s_new[j];
              }
            }

            energy = 0.0;
            for (auto it = hamiltonians_->begin() ; it != hamiltonians_->end(); ++it) {
              energy += (*it)->calculate_total_energy();
            }

            if ( energy < e_min ) {
              // this configuration is the new minimum
              e_min = energy;
              std::copy(globals::s.begin(), globals::s.end(), s_min.begin());
            }
            return false;
          }

      operator std::uint64_t() const {return count;}
  };

  void MetropolisMCSolver::SystematicPreconditioner(const double delta_theta, const double delta_phi) {
    // TODO: this should probably rotate spins rather than set definite direction so we can then handle
    // ferrimagnets too
    int num_theta;
    // double e_min, e_final, phi;

    Vec3 s_new;

    jams::MultiArray<double,2> s_init(globals::s);
    jams::MultiArray<double,2> s_min(globals::s);

    num_theta = (180.0 / delta_theta) + 1;


    std::vector<double> theta(num_theta);

    theta[0] = 0.0;
    for (int i = 1; i < num_theta; ++i) {
      theta[i] = theta[i-1] + delta_theta;
    }

    MagnetizationRotationMinimizer minimizer(hamiltonians_);

    cout << "    delta theta (deg) " << delta_theta << "\n";

    cout << "    num_theta " << num_theta << "\n";

    std::uint64_t count = for_each_permutation(theta.begin(),
                                                   theta.begin() + 3,
                                                   theta.end(),
                                                   minimizer);

    cout << "    permutations " << count << "\n";

    std::ofstream preconditioner_file(jams::output::full_path_filename("mc_pre.tsv"));
    preconditioner_file << "# theta (deg) | phi (deg) | energy (J) \n";

    preconditioner_file.close();

    // use the minimum configuration
    std::copy(minimizer.s().begin(),  minimizer.s().end(), globals::s.begin());
  }

  int MetropolisMCSolver::MetropolisAlgorithm(std::function<Vec3(Vec3)> trial_spin_move) {
    using std::min;
    using std::exp;
    std::uniform_real_distribution<> uniform_distribution;

    const double beta = kBohrMagneton / (kBoltzmann * physics_module_->temperature());

    unsigned moves_accepted = 0;
    for (auto n = 0; n < globals::num_spins; ++n) {
      // 2015-12-10 (JB) striding uniformly is ~4x faster than random choice (clang OSX).
      // Seems to be because of caching/predication in the exchange field calculation.
      int spin_index = n;

      if (use_random_spin_order_) {
        spin_index = jams::instance().random_generator()(globals::num_spins);
      }

      auto s_initial = mc_spin_as_vec(spin_index);
      auto s_final = trial_spin_move(s_initial);

      auto deltaE = 0.0;
      for (const auto& ham : hamiltonians_) {
        deltaE += ham->calculate_one_spin_energy_difference(spin_index, s_initial, s_final);
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

  const double beta = kBohrMagneton / (kBoltzmann * physics_module_->temperature());

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
