// Copyright 2014 Joseph Barker. All rights reserved.
#include <iomanip>

#include <libconfig.h++>

#include "cpu_monte_carlo_constrained.h"

#include "jams/core/jams++.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/helpers/montecarlo.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"

using namespace std;

void ConstrainedMCSolver::initialize(const libconfig::Setting& settings) {
  // initialize base class
  Solver::initialize(settings);

  max_steps_ = jams::config_required<int>(settings, "max_steps");
  min_steps_ = jams::config_optional<int>(settings, "min_steps", jams::defaults::solver_min_steps);

  constraint_theta_        = jams::config_required<double>(settings, "cmc_constraint_theta");
  constraint_phi_          = jams::config_required<double>(settings, "cmc_constraint_phi");
  move_angle_sigma_        = jams::config_optional<double>(settings, "move_angle_sigma", jams::defaults::solver_monte_carlo_move_sigma);
  output_write_steps_      = jams::config_optional<int>(settings, "output_write_steps",  jams::defaults::monitor_output_steps);

  constraint_vector_       = spherical_to_cartesian_vector(1.0, deg_to_rad(constraint_theta_), deg_to_rad(constraint_phi_));

  // from cartesian into the constraint space
  rotation_matrix_         = rotation_matrix_y(-deg_to_rad(constraint_theta_))*rotation_matrix_z(deg_to_rad(-constraint_phi_));
  // from the constraint space back to cartesian
  inverse_rotation_matrix_ = rotation_matrix_z(deg_to_rad(constraint_theta_)) * rotation_matrix_y(deg_to_rad(constraint_phi_));

  if (settings.exists("move_fraction_uniform") || settings.exists("move_fraction_angle") || settings.exists("move_fraction_reflection")) {
    move_fraction_uniform_    = jams::config_optional<double>(settings, "move_fraction_uniform", 0.0);
    move_fraction_angle_      = jams::config_optional<double>(settings, "move_fraction_angle", 0.0);
    move_fraction_reflection_ = jams::config_optional<double>(settings, "move_fraction_reflection", 0.0);

    double move_fraction_sum = move_fraction_uniform_ + move_fraction_angle_ + move_fraction_reflection_;

    move_fraction_uniform_     /= move_fraction_sum;
    move_fraction_angle_       /= move_fraction_sum;
    move_fraction_reflection_  /= move_fraction_sum;
  }


  spin_transformations_.resize(globals::num_spins);
  for (int i = 0; i < globals::num_spins; ++i) {
    spin_transformations_[i] = lattice->material(lattice->atom_material_id(i)).transform;
  }

  cout << "    constraint angle theta (deg) " << constraint_theta_ << "\n";
  cout << "    constraint angle phi (deg) " << constraint_phi_ << "\n";
  cout << "    constraint vector " << constraint_vector_[0] << " " << constraint_vector_[1] << " " << constraint_vector_[2] << "\n";
  cout << "    move_fraction_uniform " << move_fraction_uniform_ << "\n";
  cout << "    move_fraction_angle " << move_fraction_angle_ << "\n";
  cout << "    move_fraction_reflection " << move_fraction_reflection_ << "\n";
  cout << "    move_angle_sigma " << move_angle_sigma_ << "\n";
  cout << "    output_write_steps " << output_write_steps_ << "\n";
  cout << "    rotation matrix m -> mz\n";
  for (auto i = 0; i < 3; ++i) {
    cout << "      ";
    for (auto j = 0; j < 3; ++j) {
      cout << rotation_matrix_[i][j] << " ";
    }
    cout << "\n";
  }
  cout << "    inverse rotation matrix mz -> m\n";
  for (auto i = 0; i < 3; ++i) {
    cout << "      ";
    for (auto j = 0; j < 3; ++j) {
      cout << inverse_rotation_matrix_[i][j] << " ";
    }
    cout << "\n";
  }

  // do some basic checks
  if (approximately_equal(move_fraction_reflection_, 1.0, 1e-8)) {
    jams_warning("Only reflection moves have been configured. This breaks ergodicity.");
  }

  Vec3 test_unit_vec = {0.0, 0.0, 1.0};
  Vec3 test_forward_vec = rotation_matrix_ * test_unit_vec;
  Vec3 test_back_vec    = inverse_rotation_matrix_ * test_forward_vec;

  if (verbose_is_enabled()) {
    cout << "  rotation sanity check\n";
    cout << "    rotate\n";
    cout << "      " << test_unit_vec << " -> " << test_forward_vec << "\n";
    cout << "    back rotate\n";
    cout << "      " << test_forward_vec << " -> " << test_back_vec << "\n";
  }

  for (int n = 0; n < 3; ++n) {
    if (!approximately_equal(test_unit_vec[n], test_back_vec[n], jams::defaults::solver_monte_carlo_constraint_tolerance)) {
      throw std::runtime_error("ConstrainedMCSolver :: rotation sanity check failed");
    }
  }
}

void ConstrainedMCSolver::run() {
  // Chooses nspins random spin pairs from the spin system and attempts a
  // Constrained Monte Carlo move on each pair, accepting for either lower
  // energy or with a Boltzmann thermal weighting.
  std::uniform_real_distribution<> uniform_distribution;

  MonteCarloUniformMove<pcg32_k1024> uniform_move(&random_generator_);
  MonteCarloAngleMove<pcg32_k1024>   angle_move(&random_generator_, move_angle_sigma_);
  MonteCarloReflectionMove           reflection_move;

  const double uniform_random_number = uniform_distribution(random_generator_);
  if (uniform_random_number < move_fraction_uniform_) {
    move_running_acceptance_count_uniform_ += AsselinAlgorithm(uniform_move);
    run_count_uniform++;
  } else if (uniform_random_number < (move_fraction_uniform_ + move_fraction_angle_)) {
    move_running_acceptance_count_angle_ += AsselinAlgorithm(angle_move);
    run_count_angle++;
  } else {
    move_running_acceptance_count_reflection_ += AsselinAlgorithm(reflection_move);
    run_count_reflection++;
  }

  iteration_++;

  if (iteration_ % output_write_steps_ == 0) {

    validate_constraint();

    move_total_count_uniform_    += run_count_uniform;
    move_total_count_angle_      += run_count_angle;
    move_total_count_reflection_ += run_count_reflection;

    move_total_acceptance_count_uniform_    += move_running_acceptance_count_uniform_;
    move_total_acceptance_count_angle_      += move_running_acceptance_count_angle_;
    move_total_acceptance_count_reflection_ += move_running_acceptance_count_reflection_;

    cout << "\n";
    cout << "iteration" << iteration_ << "\n";
    cout << "move_acceptance_fraction\n";

    double half_num_spins = 0.5 * globals::num_spins;

    cout << "  uniform ";
    cout << division_or_zero(move_running_acceptance_count_uniform_, half_num_spins * run_count_uniform) << " (";
    cout << division_or_zero(move_total_acceptance_count_uniform_,   half_num_spins * move_total_count_uniform_) << ") \n";

    cout << "  angle ";
    cout << division_or_zero(move_running_acceptance_count_angle_, half_num_spins * run_count_angle) << " (";
    cout << division_or_zero(move_total_acceptance_count_angle_,   half_num_spins * move_total_count_angle_) << ") \n";

    cout << "  reflection ";
    cout << division_or_zero(move_running_acceptance_count_reflection_, half_num_spins * run_count_reflection) << " (";
    cout << division_or_zero(move_total_acceptance_count_reflection_,   half_num_spins * move_total_count_reflection_) << ") \n";
    
    move_running_acceptance_count_uniform_    = 0;
    move_running_acceptance_count_angle_      = 0;
    move_running_acceptance_count_reflection_ = 0;

    run_count_uniform    = 0;
    run_count_angle      = 0;
    run_count_reflection = 0;
  }
}

unsigned ConstrainedMCSolver::AsselinAlgorithm(const std::function<Vec3(Vec3)>&  trial_spin_move) {
  std::uniform_real_distribution<> uniform_distribution;

  const double    beta = kBohrMagneton / (physics_module_->temperature() * kBoltzmann);
  Vec3         m_total = total_transformed_magnetization();

  unsigned moves_accepted = 0;

  // we move two spins moving all spins on average is num_spins/2
  for (auto i = 0; i < globals::num_spins/2; ++i) {
    // randomly get two spins s1 != s2
    auto s1 = static_cast<int>(random_generator_(globals::num_spins));
    auto s2 = s1;
    while (s2 == s1) {
      s2 = static_cast<int>(random_generator_(globals::num_spins));
    }

    Vec3 s1_initial         = mc_spin_as_vec(s1);
    Vec3 s1_initial_rotated = rotate_cartesian_to_constraint(s1, s1_initial);

    Vec3 s1_trial           = trial_spin_move(s1_initial);
    Vec3 s1_trial_rotated   = rotate_cartesian_to_constraint(s1, s1_trial);

    Vec3 s2_initial         = mc_spin_as_vec(s2);
    Vec3 s2_initial_rotated = rotate_cartesian_to_constraint(s2, s2_initial);

    // calculate new spin based on contraint mx = my = 0 in the constraint vector reference frame
    Vec3 s2_trial_rotated   = s2_initial_rotated + (s1_initial_rotated - s1_trial_rotated ) * (globals::mus(s1) / globals::mus(s2)) ;

    double ss2 = s2_trial_rotated[0] * s2_trial_rotated[0] + s2_trial_rotated[1] * s2_trial_rotated[1];
    if (ss2 > 1.0) {
      // the rotated spin does not fit on the unit sphere - revert s1 and reject move
      continue;
    }
    // calculate the z-component so that |s2| = 1
    s2_trial_rotated[2] = std::copysign(sqrt(1.0 - ss2), s2_initial_rotated[2]);

    Vec3 s2_trial = rotate_constraint_to_cartesian(s2, s2_trial_rotated);

    Vec3 deltaM = magnetization_difference(s1, s1_initial, s1_trial, s2, s2_initial, s2_trial);

    Vec3 m_trial_rotated = rotation_matrix_ * (m_total + deltaM);
    if (m_trial_rotated[2] < 0.0) {
      // The new magnetization is in the opposite sense - revert s1, reject move
      continue;
    }
    Vec3 m_initial_rotated = rotation_matrix_ * (m_total);

    double deltaE = energy_difference(s1, s1_initial, s1_trial, s2, s2_initial, s2_trial);

    // calculate the Boltzmann weighted probability including the Jacobian factors (see paper)
    double probability = min(1.0, exp(-deltaE * beta) * pow2(m_trial_rotated[2] / m_initial_rotated[2]) * std::abs(s2_initial_rotated[2] / s2_trial_rotated[2]));

    auto x = uniform_distribution(random_generator_);
    if (x > probability) {
      // reject move
      continue;
    }

    // accept move
    mc_set_spin_as_vec(s1, s1_trial);
    mc_set_spin_as_vec(s2, s2_trial);

    m_total += deltaM;

    moves_accepted++;
  }

  return moves_accepted;
}

double ConstrainedMCSolver::energy_difference(const int s1, const Vec3 &s1_initial, const Vec3 &s1_trial,
                                              const int s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const {
  double delta_energy1 = 0.0;
  for (auto hamiltonian : hamiltonians_) {
    delta_energy1 += hamiltonian->calculate_one_spin_energy_difference(s1, s1_initial, s1_trial);
  }

  // temporarily accept the move for s1 so we can calculate the s2 energies
  mc_set_spin_as_vec(s1, s1_trial);
  double delta_energy2 = 0.0;
  for (auto hamiltonian : hamiltonians_) {
    delta_energy2 += hamiltonian->calculate_one_spin_energy_difference(s2, s2_initial, s2_trial);
  }
  mc_set_spin_as_vec(s1, s1_initial);

  return delta_energy1 + delta_energy2;
}

Vec3 ConstrainedMCSolver::magnetization_difference(const int s1, const Vec3 &s1_initial, const Vec3 &s1_trial,
                                                   const int s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const {
  return globals::mus(s1) * spin_transformations_[s1] * (s1_trial - s1_initial)
  + globals::mus(s2) * spin_transformations_[s2] * (s2_trial - s2_initial);
}

Vec3 ConstrainedMCSolver::total_transformed_magnetization() const {
  Vec3 m_total = {0.0, 0.0, 0.0};

  for (auto i = 0; i < globals::num_spins; ++i) {
    m_total += globals::mus(i) * spin_transformations_[i] * mc_spin_as_vec(i);
  }

  return m_total;
}

void ConstrainedMCSolver::validate_constraint() const {
    Vec3 m_total = total_transformed_magnetization();

   if (!approximately_equal(rad_to_deg(polar_angle(m_total)), constraint_theta_, jams::defaults::solver_monte_carlo_constraint_tolerance)) {
     std::stringstream ss;
     ss << "ConstrainedMCSolver::AsselinAlgorithm -- theta constraint violated (" << rad_to_deg(polar_angle(m_total)) << " deg)";
     throw std::runtime_error(ss.str());
   }

   if (!approximately_equal(rad_to_deg(azimuthal_angle(m_total)), constraint_phi_, jams::defaults::solver_monte_carlo_constraint_tolerance)) {
     std::stringstream ss;
     ss << "ConstrainedMCSolver::AsselinAlgorithm -- phi constraint violated (" << rad_to_deg(azimuthal_angle(m_total)) << " deg)";
   }
}

Vec3 ConstrainedMCSolver::rotate_cartesian_to_constraint(const int i, const Vec3 &spin) const {
  return spin_transformations_[i] * rotation_matrix_ * spin;
}

Vec3 ConstrainedMCSolver::rotate_constraint_to_cartesian(const int i, const Vec3 &spin) const {
  return transpose(spin_transformations_[i]) * inverse_rotation_matrix_ * spin;
}


