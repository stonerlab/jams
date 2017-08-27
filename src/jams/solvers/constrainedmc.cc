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

ConstrainedMCSolver::~ConstrainedMCSolver() {
  if (outfile.is_open()) {
    outfile.close();
  }
}

void ConstrainedMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

    // initialize base class
  Solver::initialize(argc, argv, idt);

  output->write("\n----------------------------------------\n");
  output->write("\nConstrained Monte-Carlo solver\n");

  libconfig::Setting &solver_settings = ::config->lookup("solver");

  solver_settings.lookupValue("output_write_steps", output_write_steps_);

  configure_move_types(solver_settings);

  ::output->write("move_fraction_uniform:     % 8.4f\n", move_fraction_uniform_);
  ::output->write("move_fraction_angle:       % 8.4f\n", move_fraction_angle_);
  ::output->write("move_fraction_reflection:  % 8.4f\n", move_fraction_reflection_);
  ::output->write("\n");
  ::output->write("move_angle_sigma:          % 8.4f\n", move_angle_sigma_);

  constraint_theta_ = solver_settings["cmc_constraint_theta"];
  constraint_phi_   = solver_settings["cmc_constraint_phi"];
  constraint_vector_ = cartesian_from_spherical(1.0, constraint_theta_, constraint_phi_);

  ::output->write("  constraint angle theta (deg)\n    % 8.8f\n", constraint_theta_);
  ::output->write("  constraint angle phi (deg)\n    % 8.8f\n", constraint_phi_);
  ::output->write("  constraint vector\n    % 8.8f, % 8.8f, % 8.8f\n", constraint_vector_[0], constraint_vector_[1], constraint_vector_[2]);

  // calculate rotation matrix for rotating m -> mz
  Mat3 r_y = create_rotation_matrix_y(constraint_theta_);
  Mat3 r_z = create_rotation_matrix_z(constraint_phi_);

  inverse_rotation_matrix_ = r_y * r_z;
  rotation_matrix_ = transpose(inverse_rotation_matrix_);

  ::output->verbose("  Rot_y matrix\n");
  ::output->verbose("    % 8.8f  % 8.8f  % 8.8f\n", r_y[0][0], r_y[0][1], r_y[0][2]);
  ::output->verbose("    % 8.8f  % 8.8f  % 8.8f\n", r_y[1][0], r_y[1][1], r_y[1][2]);
  ::output->verbose("    % 8.8f  % 8.8f  % 8.8f\n", r_y[2][0], r_y[2][1], r_y[2][2]);

  ::output->verbose("  Rot_z matrix\n");
  ::output->verbose("    % 8.8f  % 8.8f  % 8.8f\n", r_z[0][0], r_z[0][1], r_z[0][2]);
  ::output->verbose("    % 8.8f  % 8.8f  % 8.8f\n", r_z[1][0], r_z[1][1], r_z[1][2]);
  ::output->verbose("    % 8.8f  % 8.8f  % 8.8f\n", r_z[2][0], r_z[2][1], r_z[2][2]);

  ::output->write("  rotation matrix m -> mz\n");
  ::output->write("    % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[0][0], rotation_matrix_[0][1], rotation_matrix_[0][2]);
  ::output->write("    % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[1][0], rotation_matrix_[1][1], rotation_matrix_[1][2]);
  ::output->write("    % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[2][0], rotation_matrix_[2][1], rotation_matrix_[2][2]);

  ::output->write("  inverse rotation matrix mz -> m\n");
  ::output->write("    % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[0][0], inverse_rotation_matrix_[0][1], inverse_rotation_matrix_[0][2]);
  ::output->write("    % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[1][0], inverse_rotation_matrix_[1][1], inverse_rotation_matrix_[1][2]);
  ::output->write("    % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[2][0], inverse_rotation_matrix_[2][1], inverse_rotation_matrix_[2][2]);

  // --- sanity check
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

  // create spin transform arrays
  s_transform_.resize(num_spins);

  for (int i = 0; i < num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      for (int m = 0; m < 3; ++m) {
        s_transform_(i)[n][m] = 0.0;
      }
    }
  }

  libconfig::Setting& material_settings = ::config->lookup("materials");
  for (int i = 0; i < num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      s_transform_(i)[n][n] = material_settings[::lattice->atom_material(i)]["transform"][n];
    }
  }

  outfile.open(std::string(::seedname + "_mc_stats.dat").c_str());
}

void ConstrainedMCSolver::configure_move_types(const libconfig::Setting& config) {

  if (config.exists("move_fraction_uniform") || config.exists("move_fraction_angle")
     || config.exists("move_fraction_reflection_")) {
    move_fraction_uniform_    = 0.0;
    move_fraction_angle_      = 0.0;
    move_fraction_reflection_ = 0.0;

    config.lookupValue("move_fraction_uniform",    move_fraction_uniform_);
    config.lookupValue("move_fraction_angle",      move_fraction_angle_);
    config.lookupValue("move_fraction_reflection", move_fraction_reflection_);
  }

  const double move_fraction_sum = move_fraction_uniform_ + move_fraction_angle_ + move_fraction_reflection_;

  move_fraction_uniform_     /= move_fraction_sum;
  move_fraction_angle_       /= move_fraction_sum;
  move_fraction_reflection_  /= move_fraction_sum;

  if (floats_are_equal(move_fraction_reflection_, 1.0)) {
    jams_warning("Only reflection moves have been configured. This breaks ergodicity.");
  }

  if (config.exists("move_angle_sigma")) {
    config.lookupValue("move_angle_sigma", move_angle_sigma_);
  }

}

void ConstrainedMCSolver::run() {
  // Chooses nspins random spin pairs from the spin system and attempts a
  // Constrained Monte Carlo move on each pair, accepting for either lower
  // energy or with a Boltzmann thermal weighting.
  using namespace globals;

  std::uniform_real_distribution<> uniform_distribution;

  MonteCarloUniformMove<pcg32> uniform_move(&random_generator_);
  MonteCarloAngleMove<pcg32> angle_move(&random_generator_, move_angle_sigma_);
  MonteCarloReflectionMove reflection_move;

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
    move_total_count_uniform_    += run_count_uniform;
    move_total_count_angle_      += run_count_angle;
    move_total_count_reflection_ += run_count_reflection;

    move_total_acceptance_count_uniform_ += move_running_acceptance_count_uniform_;
    move_total_acceptance_count_angle_ += move_running_acceptance_count_angle_;
    move_total_acceptance_count_reflection_ += move_running_acceptance_count_reflection_;

    ::output->write("\n");
    ::output->write("iteration: %d\n", iteration_);
    ::output->write("move_acceptance_fraction\n");

    ::output->write("  uniform:    %4.4f (%4.4f)\n",
                    division_or_zero(move_running_acceptance_count_uniform_, (0.5 * num_spins * run_count_uniform)),
                    division_or_zero(move_total_acceptance_count_uniform_, (0.5 * num_spins * move_total_count_uniform_)));

    ::output->write("  angle:      %4.4f (%4.4f)\n",
                    division_or_zero(move_running_acceptance_count_angle_, (0.5 * num_spins * run_count_angle)),
                    division_or_zero(move_total_acceptance_count_angle_, (0.5 * num_spins * move_total_count_angle_)));

    ::output->write("  reflection: %4.4f (%4.4f)\n",
                    division_or_zero(move_running_acceptance_count_reflection_, (0.5 * num_spins * run_count_reflection)),
                    division_or_zero(move_total_acceptance_count_reflection_, (0.5 * num_spins * move_total_count_reflection_)));

    move_running_acceptance_count_uniform_    = 0;
    move_running_acceptance_count_angle_      = 0;
    move_running_acceptance_count_reflection_ = 0;

    run_count_uniform    = 0;
    run_count_angle      = 0;
    run_count_reflection = 0;
  }
}

unsigned ConstrainedMCSolver::AsselinAlgorithm(std::function<Vec3(Vec3)>  move) {
  std::uniform_real_distribution<> uniform_distribution;
  std::uniform_int_distribution<> uniform_int_distribution(0, globals::num_spins-1);

  int rand_s1, rand_s2;
  double delta_energy1, delta_energy2, delta_energy21;
  double mu1, mu2, mz_old, mz_new, probability;
  const double beta = kBohrMagneton/(physics_module_->temperature()*kBoltzmann);

  Vec3 s1_initial, s1_final, s1_initial_rotated, s1_final_rotated;
  Vec3 s2_initial, s2_final, s2_initial_rotated, s2_final_rotated;

  Mat3 s1_transform, s1_transform_inv;
  Mat3 s2_transform, s2_transform_inv;

  Vec3 m_other = {0.0, 0.0, 0.0};

  for (int i = 0; i < globals::num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      m_other[n] += s_transform_(i)[n][n] * globals::s(i,n) * globals::mus(i);
    }
  }

  // if (!floats_are_equal(rad_to_deg(azimuthal_angle(m_other)), constraint_theta_)) {
  //   std::stringstream ss;
  //   ss << "ConstrainedMCSolver::AsselinAlgorithm -- theta constraint violated (" << rad_to_deg(azimuthal_angle(m_other)) << " deg)";
  //   throw std::runtime_error(ss.str());
  // }

  // if (!floats_are_equal(rad_to_deg(polar_angle(m_other)), constraint_phi_)) {
  //   std::stringstream ss;
  //   ss << "ConstrainedMCSolver::AsselinAlgorithm -- phi constraint violated (" << rad_to_deg(polar_angle(m_other)) << " deg)";
  // }

  unsigned moves_accepted = 0;

  // In this 'for' loop I've tried to bail and continue at the earliest possible point
  // in all cases to try and avoid extra unneccesary calculations when the move will be
  // rejected.

  // we move two spins moving all spins on average is num_spins/2
  for (int i = 0; i < globals::num_spins/2; ++i) {

    // randomly get two spins s1 != s2
    rand_s1 = uniform_int_distribution(random_generator_);
    rand_s2 = rand_s1;
    while (rand_s2 == rand_s1) {
      rand_s2 = uniform_int_distribution(random_generator_);
    }

    s1_transform = s_transform_(rand_s1);
    s2_transform = s_transform_(rand_s2);

    s1_initial = mc_spin_as_vec(rand_s1);
    s2_initial = mc_spin_as_vec(rand_s2);

    s1_transform_inv = transpose(s1_transform);
    s2_transform_inv = transpose(s2_transform);

    mu1 = globals::mus(rand_s1);
    mu2 = globals::mus(rand_s2);

    // rotate into reference frame of the constraint vector
    s1_initial_rotated = s1_transform * rotation_matrix_ * s1_initial;
    s2_initial_rotated = s2_transform * rotation_matrix_ * s2_initial;

    // Monte Carlo move
    s1_final = s1_initial;
    s1_final = move(s1_final);
    s1_final_rotated = s1_transform * rotation_matrix_*s1_final;

    // calculate new spin based on contraint mx = my = 0 in the constraint vector reference frame
    s2_final_rotated = ((s1_initial_rotated - s1_final_rotated ) * (mu1/mu2) 
                      + s2_initial_rotated);

    // zero out the z-component which will be calculated below
    s2_final_rotated[2] = 0.0;

    if (unlikely(dot(s2_final_rotated, s2_final_rotated) > 1.0)) {
      // the rotated spin does not fit on the unit sphere - revert s1 and reject move
      continue;
    }

    // calculate the z-component so that |s2| = 1
    s2_final_rotated[2] = std::copysign(1.0, s2_initial_rotated[2]) * sqrt(1.0 - dot(s2_final_rotated, s2_final_rotated));

    // rotate s2 back into the cartesian reference frame
    s2_final = s2_transform_inv*inverse_rotation_matrix_*s2_final_rotated;

    mz_new = dot((m_other + 
      s1_transform*s1_final*mu1 + s2_transform*s2_final*mu2 
      - s1_transform*s1_initial*mu1 - s2_transform*s2_initial*mu2), constraint_vector_);

    // The new magnetization is in the opposite sense - revert s1, reject move
    if (unlikely(mz_new < 0.0)) {
      continue;
    }

    // change in energy with spin move
    delta_energy1 = 0.0;

    for (auto it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
      delta_energy1 += (*it)->calculate_one_spin_energy_difference(rand_s1, s1_initial, s1_final);
    }

    // temporarily accept the move for s1 so we can calculate the s2 energies
    // this will be reversed later if the move is rejected
    mc_set_spin_as_vec(rand_s1, s1_final);

    delta_energy2 = 0.0;
    for (auto it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
      delta_energy2 += (*it)->calculate_one_spin_energy_difference(rand_s2, s2_initial, s2_final);
    }

    // calculate the total energy difference
    delta_energy21 = delta_energy1 + delta_energy2;

    mz_old = dot(m_other, constraint_vector_);

    // calculate the Boltzmann weighted probability including the Jacobian factors (see paper)
    probability = exp(-delta_energy21*beta)*(pow2(mz_new/mz_old))*std::abs(s2_initial_rotated[2]/s2_final_rotated[2]);

    if (probability < uniform_distribution(random_generator_)) {
      // move fails to overcome Boltzmann factor - revert s1, reject move
      mc_set_spin_as_vec(rand_s1, s1_initial);
      continue;
    } else {
      // accept move
      mc_set_spin_as_vec(rand_s2, s2_final);
      m_other += s1_transform*s1_final*mu1 + s2_transform*s2_final*mu2 
      - s1_transform*s1_initial*mu1 - s2_transform*s2_initial*mu2;
      moves_accepted++;
    }
  }

  return moves_accepted;
}
