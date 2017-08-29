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
  constraint_vector_ = cartesian_from_spherical(1.0, deg_to_rad(constraint_theta_), deg_to_rad(constraint_phi_));

  ::output->write("  constraint angle theta (deg)\n    % 8.8f\n", constraint_theta_);
  ::output->write("  constraint angle phi (deg)\n    % 8.8f\n", constraint_phi_);
  ::output->write("  constraint vector\n    % 8.8f, % 8.8f, % 8.8f\n", constraint_vector_[0], constraint_vector_[1], constraint_vector_[2]);

  // calculate rotation matrix for rotating m -> mz
  Mat3 r_y = create_rotation_matrix_y(deg_to_rad(constraint_theta_));
  Mat3 r_z = create_rotation_matrix_z(deg_to_rad(constraint_phi_));

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

  MonteCarloUniformMove<pcg64_k1024> uniform_move(&random_generator_);
  MonteCarloAngleMove<pcg64_k1024> angle_move(&random_generator_, move_angle_sigma_);
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

    validate_constraint();

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

unsigned ConstrainedMCSolver::AsselinAlgorithm(std::function<Vec3(Vec3)>  trial_spin_move) {
  std::uniform_real_distribution<> uniform_distribution;

  double mz_old, mz_new;
  const double beta = kBohrMagneton / (physics_module_->temperature() * kBoltzmann);

  Vec3 s1_initial, s1_final, s1_initial_rotated, s1_final_rotated;
  Vec3 s2_initial, s2_final, s2_initial_rotated, s2_final_rotated;

  Mat3 s1_transform, s1_transform_inv;
  Mat3 s2_transform, s2_transform_inv;

  Vec3 m_total = total_transformed_magnetization();

  unsigned moves_accepted = 0;

  // we move two spins moving all spins on average is num_spins/2
  for (auto i = 0; i < globals::num_spins/2; ++i) {

    // randomly get two spins s1 != s2
    auto rand_s1 = random_generator_(globals::num_spins);
    auto rand_s2 = rand_s1;
    while (rand_s2 == rand_s1) {
      rand_s2 = random_generator_(globals::num_spins);
    }

    double mu1 = globals::mus(rand_s1);
    double mu2 = globals::mus(rand_s2);

    s1_transform = s_transform_(rand_s1);
    s2_transform = s_transform_(rand_s2);

    s1_initial = mc_spin_as_vec(rand_s1);
    s2_initial = mc_spin_as_vec(rand_s2);

    s2_transform_inv = transpose(s2_transform);

    // rotate into reference frame of the constraint vector
    s1_initial_rotated = s1_transform * rotation_matrix_ * s1_initial;
    s2_initial_rotated = s2_transform * rotation_matrix_ * s2_initial;

    // Monte Carlo move
    s1_final = trial_spin_move(s1_initial);
    s1_final_rotated = s1_transform * rotation_matrix_*s1_final;

    // calculate new spin based on contraint mx = my = 0 in the constraint vector reference frame
    s2_final_rotated = ((s1_initial_rotated - s1_final_rotated ) * (mu1/mu2) + s2_initial_rotated);

    const double ss2 = s2_final_rotated[0] * s2_final_rotated[0] + s2_final_rotated[1] * s2_final_rotated[1];

    if (ss2 > 1.0) {
      // the rotated spin does not fit on the unit sphere - revert s1 and reject move
      continue;
    }

    // calculate the z-component so that |s2| = 1
    s2_final_rotated[2] = std::copysign(sqrt(1.0 - ss2), s2_initial_rotated[2]);

    // rotate s2 back into the cartesian reference frame
    s2_final = s2_transform_inv*inverse_rotation_matrix_*s2_final_rotated;

    const Vec3 deltaM = magnetization_difference(rand_s1, s1_initial, s1_final, rand_s2, s2_initial, s2_final);

    mz_new = dot(m_total + deltaM, constraint_vector_);

    if (mz_new < 0.0) {
      // The new magnetization is in the opposite sense - revert s1, reject move
      continue;
    }

    mz_old = dot(m_total, constraint_vector_);

    const double deltaE = energy_difference(rand_s1, s1_initial, s1_final, rand_s2, s2_initial, s2_final);

    // calculate the Boltzmann weighted probability including the Jacobian factors (see paper)
    double probability = exp(-deltaE * beta)*(pow2(mz_new/mz_old))*std::abs(s2_initial_rotated[2]/s2_final_rotated[2]);

    if (probability < uniform_distribution(random_generator_)) {
      // reject move
      continue;
    } else {
      // accept move
      mc_set_spin_as_vec(rand_s1, s1_final);
      mc_set_spin_as_vec(rand_s2, s2_final);

      m_total += deltaM;

      moves_accepted++;
    }
  }

  return moves_accepted;
}

double ConstrainedMCSolver::energy_difference(unsigned rand_s1, const Vec3 &s1_initial, const Vec3 &s1_final,
                                              unsigned rand_s2, const Vec3 &s2_initial, const Vec3 &s2_final) const {
  double delta_energy1 = 0.0;

  for (auto it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    delta_energy1 += (*it)->calculate_one_spin_energy_difference(rand_s1, s1_initial, s1_final);
  }

  // temporarily accept the move for s1 so we can calculate the s2 energies
  // this will be reversed later if the move is rejected
  mc_set_spin_as_vec(rand_s1, s1_final);

  double delta_energy2 = 0.0;
  for (auto it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    delta_energy2 += (*it)->calculate_one_spin_energy_difference(rand_s2, s2_initial, s2_final);
  }

  mc_set_spin_as_vec(rand_s1, s1_initial);

  return delta_energy1 + delta_energy2;
}

Vec3 ConstrainedMCSolver::magnetization_difference(unsigned rand_s1, const Vec3 &s1_initial, const Vec3 &s1_final,
                                                   unsigned rand_s2, const Vec3 &s2_initial, const Vec3 &s2_final) const {
  return globals::mus(rand_s1) * s_transform_(rand_s1) * (s1_final - s1_initial)
  + globals::mus(rand_s2) * s_transform_(rand_s2) * (s2_final - s2_initial);
}

Vec3 ConstrainedMCSolver::total_transformed_magnetization() const {
  Vec3 m_total = {0.0, 0.0, 0.0};

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto n = 0; n < 3; ++n) {
      m_total[n] += s_transform_(i)[n][n] * globals::s(i,n) * globals::mus(i);
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