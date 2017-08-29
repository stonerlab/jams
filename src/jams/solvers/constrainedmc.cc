// Copyright 2014 Joseph Barker. All rights reserved.
#include <iomanip>

#include <libconfig.h++>
#include <jams/core/jams++.h>

#include "jams/solvers/constrainedmc.h"

#include "jams/core/utils.h"
#include "jams/core/consts.h"
#include "jams/core/maths.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/montecarlo.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"

namespace {
    double division_or_zero(const double nominator, const double denominator) {
      if (denominator == 0.0) {
        return 0.0;
      } else {
        return nominator / denominator;
      }
    }
}

void ConstrainedMCSolver::initialize(int argc, char **argv, double idt) {
  // initialize base class
  Solver::initialize(argc, argv, idt);

  libconfig::Setting &cfg = ::config->lookup("solver");

  constraint_theta_        = cfg["cmc_constraint_theta"];
  constraint_phi_          = cfg["cmc_constraint_phi"];
  constraint_vector_       = cartesian_from_spherical(1.0, deg_to_rad(constraint_theta_), deg_to_rad(constraint_phi_));
  inverse_rotation_matrix_ = rotation_matrix_y(deg_to_rad(constraint_theta_)) * rotation_matrix_z(deg_to_rad(constraint_phi_));
  rotation_matrix_         = transpose(inverse_rotation_matrix_);

  if (cfg.exists("move_fraction_uniform") || cfg.exists("move_fraction_angle") || cfg.exists("move_fraction_reflection")) {
    move_fraction_uniform_    = 0.0;
    move_fraction_angle_      = 0.0;
    move_fraction_reflection_ = 0.0;

    cfg.lookupValue("move_fraction_uniform",    move_fraction_uniform_);
    cfg.lookupValue("move_fraction_angle",      move_fraction_angle_);
    cfg.lookupValue("move_fraction_reflection", move_fraction_reflection_);

    double move_fraction_sum = move_fraction_uniform_ + move_fraction_angle_ + move_fraction_reflection_;

    move_fraction_uniform_     /= move_fraction_sum;
    move_fraction_angle_       /= move_fraction_sum;
    move_fraction_reflection_  /= move_fraction_sum;
  }

  cfg.lookupValue("move_angle_sigma",   move_angle_sigma_);
  cfg.lookupValue("output_write_steps", output_write_steps_);

  // create spin transform arrays
  spin_transformations_.resize(globals::num_spins);

  libconfig::Setting& material_settings = ::config->lookup("materials");
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto m = 0; m < 3; ++m) {
      for (auto n = 0; n < 3; ++n) {
        if (m == n) {
          spin_transformations_[i][m][n] = material_settings[::lattice->atom_material(i)]["transform"][n];
        } else {
          spin_transformations_[i][m][n] = 0.0;
        }
      }
    }
  }

  ::output->write("\n----------------------------------------\n");
  ::output->write("\nConstrained Monte-Carlo solver\n");
  ::output->write("    constraint angle theta (deg)\n    % 4.4f\n", constraint_theta_);
  ::output->write("    constraint angle phi (deg)\n    % 4.4f\n", constraint_phi_);
  ::output->write("    constraint vector\n    % 8.8f, % 8.8f, % 8.8f\n", constraint_vector_[0], constraint_vector_[1], constraint_vector_[2]);
  ::output->write("\n");
  ::output->write("    move_fraction_uniform:     % 8.4f\n", move_fraction_uniform_);
  ::output->write("    move_fraction_angle:       % 8.4f\n", move_fraction_angle_);
  ::output->write("    move_fraction_reflection:  % 8.4f\n", move_fraction_reflection_);
  ::output->write("    move_angle_sigma:          % 8.4f\n", move_angle_sigma_);
  ::output->write("\n");
  ::output->write("    output_write_steps:          % 8d\n", output_write_steps_);
  ::output->write("\n");
  ::output->write("    rotation matrix m -> mz\n");
  ::output->write("      % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[0][0], rotation_matrix_[0][1], rotation_matrix_[0][2]);
  ::output->write("      % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[1][0], rotation_matrix_[1][1], rotation_matrix_[1][2]);
  ::output->write("      % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[2][0], rotation_matrix_[2][1], rotation_matrix_[2][2]);

  ::output->write("    inverse rotation matrix mz -> m\n");
  ::output->write("      % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[0][0], inverse_rotation_matrix_[0][1], inverse_rotation_matrix_[0][2]);
  ::output->write("      % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[1][0], inverse_rotation_matrix_[1][1], inverse_rotation_matrix_[1][2]);
  ::output->write("      % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[2][0], inverse_rotation_matrix_[2][1], inverse_rotation_matrix_[2][2]);

  // do some basic checks
  if (floats_are_equal(move_fraction_reflection_, 1.0)) {
    jams_warning("Only reflection moves have been configured. This breaks ergodicity.");
  }

  Vec3 test_unit_vec = {0.0, 0.0, 1.0};
  Vec3 test_forward_vec = rotation_matrix_ * test_unit_vec;
  Vec3 test_back_vec    = inverse_rotation_matrix_ * test_forward_vec;

  ::output->verbose("  rotation sanity check\n");
  ::output->verbose("    rotate\n      %f  %f  %f -> %f  %f  %f\n", test_unit_vec[0], test_unit_vec[1], test_unit_vec[2], test_forward_vec[0], test_forward_vec[1], test_forward_vec[2]);
  ::output->verbose("    back rotate\n      %f  %f  %f -> %f  %f  %f\n", test_forward_vec[0], test_forward_vec[1], test_forward_vec[2], test_back_vec[0], test_back_vec[1], test_back_vec[2]);

  for (int n = 0; n < 3; ++n) {
    if (!floats_are_equal(test_unit_vec[n], test_back_vec[n])) {
      throw std::runtime_error("ConstrainedMCSolver :: rotation sanity check failed");
    }
  }
}

void ConstrainedMCSolver::run() {
  // Chooses nspins random spin pairs from the spin system and attempts a
  // Constrained Monte Carlo move on each pair, accepting for either lower
  // energy or with a Boltzmann thermal weighting.
  std::uniform_real_distribution<> uniform_distribution;

  MonteCarloUniformMove<pcg64_k1024> uniform_move(&random_generator_);
  MonteCarloAngleMove<pcg64_k1024>   angle_move(&random_generator_, move_angle_sigma_);
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

    ::output->write("\n");
    ::output->write("iteration: %d\n", iteration_);
    ::output->write("move_acceptance_fraction\n");

    double half_num_spins = 0.5 * globals::num_spins;

    ::output->write("  uniform:    %4.4f (%4.4f)\n",
                    division_or_zero(move_running_acceptance_count_uniform_, half_num_spins * run_count_uniform),
                    division_or_zero(move_total_acceptance_count_uniform_,   half_num_spins * move_total_count_uniform_));

    ::output->write("  angle:      %4.4f (%4.4f)\n",
                    division_or_zero(move_running_acceptance_count_angle_, half_num_spins * run_count_angle),
                    division_or_zero(move_total_acceptance_count_angle_,   half_num_spins * move_total_count_angle_));

    ::output->write("  reflection: %4.4f (%4.4f)\n",
                    division_or_zero(move_running_acceptance_count_reflection_, half_num_spins * run_count_reflection),
                    division_or_zero(move_total_acceptance_count_reflection_,   half_num_spins * move_total_count_reflection_));

    move_running_acceptance_count_uniform_    = 0;
    move_running_acceptance_count_angle_      = 0;
    move_running_acceptance_count_reflection_ = 0;

    run_count_uniform    = 0;
    run_count_angle      = 0;
    run_count_reflection = 0;
  }
}

unsigned ConstrainedMCSolver::AsselinAlgorithm(std::function<Vec3(Vec3)>  trial_spin_move) {
  std::uniform_real_distribution<> uniform_distribution;

  const double    beta = kBohrMagneton / (physics_module_->temperature() * kBoltzmann);
  Vec3         m_total = total_transformed_magnetization();

  unsigned moves_accepted = 0;

  // we move two spins moving all spins on average is num_spins/2
  for (auto i = 0; i < globals::num_spins/2; ++i) {
    // randomly get two spins s1 != s2
    unsigned s1 = static_cast<unsigned int>(random_generator_(globals::num_spins));
    unsigned s2 = s1;
    while (s2 == s1) {
      s2 = static_cast<unsigned int>(random_generator_(globals::num_spins));
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

    double mz_new = dot(m_total + deltaM, constraint_vector_);
    if (mz_new < 0.0) {
      // The new magnetization is in the opposite sense - revert s1, reject move
      continue;
    }
    double mz_old = dot(m_total, constraint_vector_);

    double deltaE = energy_difference(s1, s1_initial, s1_trial, s2, s2_initial, s2_trial);

    // calculate the Boltzmann weighted probability including the Jacobian factors (see paper)
    double probability = exp(-deltaE * beta) * pow2(mz_new/mz_old) * std::abs(s2_initial_rotated[2]/s2_trial_rotated[2]);

    if (probability < uniform_distribution(random_generator_)) {
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

double ConstrainedMCSolver::energy_difference(unsigned s1, const Vec3 &s1_initial, const Vec3 &s1_trial,
                                              unsigned s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const {
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

Vec3 ConstrainedMCSolver::magnetization_difference(unsigned s1, const Vec3 &s1_initial, const Vec3 &s1_trial,
                                                   unsigned s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const {
  return globals::mus(s1) * spin_transformations_[s1] * (s1_trial - s1_initial)
  + globals::mus(s2) * spin_transformations_[s2] * (s2_trial - s2_initial);
}

Vec3 ConstrainedMCSolver::total_transformed_magnetization() const {
  Vec3 m_total = {0.0, 0.0, 0.0};

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto n = 0; n < 3; ++n) {
      m_total[n] += spin_transformations_[i][n][n] * globals::s(i,n) * globals::mus(i);
    }
  }

  return m_total;
}

void ConstrainedMCSolver::validate_constraint() const {
    Vec3 m_total = total_transformed_magnetization();

   if (!floats_are_equal(rad_to_deg(azimuthal_angle(m_total)), constraint_theta_)) {
     std::stringstream ss;
     ss << "ConstrainedMCSolver::AsselinAlgorithm -- theta constraint violated (" << rad_to_deg(azimuthal_angle(m_total)) << " deg)";
     throw std::runtime_error(ss.str());
   }

   if (!floats_are_equal(rad_to_deg(polar_angle(m_total)), constraint_phi_)) {
     std::stringstream ss;
     ss << "ConstrainedMCSolver::AsselinAlgorithm -- phi constraint violated (" << rad_to_deg(polar_angle(m_total)) << " deg)";
   }
}

Vec3 ConstrainedMCSolver::rotate_cartesian_to_constraint(unsigned i, const Vec3 &spin) const {
  return spin_transformations_[i] * rotation_matrix_ * spin;
}

Vec3 ConstrainedMCSolver::rotate_constraint_to_cartesian(unsigned i, const Vec3 &spin) const {
  return transpose(spin_transformations_[i]) * inverse_rotation_matrix_ * spin;
}


