// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/constrainedmc.h"

#include "core/utils.h"
#include "core/consts.h"
#include "core/maths.h"
#include "core/globals.h"
#include "core/hamiltonian.h"
#include "core/montecarlo.h"

#include <iomanip>

ConstrainedMCSolver::~ConstrainedMCSolver() {
  if (outfile.is_open()) {
    outfile.close();
  }
}

void ConstrainedMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

    // initialize base class
  Solver::initialize(argc, argv, idt);

  move_acceptance_count_ = 0;
  move_acceptance_fraction_ = 0.234;
  move_sigma_ = 0.05;

  output.write("Initialising Constrained Monte-Carlo solver\n");

  libconfig::Setting &solver_settings = ::config.lookup("sim");

  if (solver_settings.exists("sigma")) {
    move_sigma_ = solver_settings["sigma"];
  }

  ::output.write("\nmove sigma: % 8.8f\n", move_sigma_);

  constraint_theta_ = solver_settings["cmc_constraint_theta"];
  constraint_phi_   = solver_settings["cmc_constraint_phi"];

  ::output.write("\nconstraint angle theta (deg): % 8.8f\n", constraint_theta_);
  ::output.write("\nconstraint angle phi (deg): % 8.8f\n", constraint_phi_);

  const double c_t = cos(deg_to_rad(constraint_theta_));
  const double c_p = cos(deg_to_rad(constraint_phi_));
  const double s_t = sin(deg_to_rad(constraint_theta_));
  const double s_p = sin(deg_to_rad(constraint_phi_));

  constraint_vector_.x = s_t*c_p;
  constraint_vector_.y = s_t*s_p;
  constraint_vector_.z = c_t;

  ::output.write("\nconstraint vector: % 8.8f, % 8.8f, % 8.8f\n", constraint_vector_.x, constraint_vector_.y, constraint_vector_.z);

  // calculate rotation matrix for rotating m -> mz
  jblib::Matrix<double, 3, 3> r_y;
  jblib::Matrix<double, 3, 3> r_z;

  // first index is row second index is col
  r_y[0][0] =  c_t;  r_y[0][1] =  0.0; r_y[0][2] =  s_t;
  r_y[1][0] =  0.0;  r_y[1][1] =  1.0; r_y[1][2] =  0.0;
  r_y[2][0] = -s_t;  r_y[2][1] =  0.0; r_y[2][2] =  c_t;

  r_z[0][0] =  c_p;  r_z[0][1] = -s_p;  r_z[0][2] =  0.0;
  r_z[1][0] =  s_p;  r_z[1][1] =  c_p;  r_z[1][2] =  0.0;
  r_z[2][0] =  0.0;  r_z[2][1] =  0.0;  r_z[2][2] =  1.0;

  inverse_rotation_matrix_ = r_y*r_z;
  rotation_matrix_ = inverse_rotation_matrix_.transpose();

  ::output.write("\nRy\n");
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", r_y[0][0], r_y[0][1], r_y[0][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", r_y[1][0], r_y[1][1], r_y[1][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", r_y[2][0], r_y[2][1], r_y[2][2]);

  ::output.write("\nRz\n");
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", r_z[0][0], r_z[0][1], r_z[0][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", r_z[1][0], r_z[1][1], r_z[1][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", r_z[2][0], r_z[2][1], r_z[2][2]);

  ::output.write("\nrotation matrix m -> mz\n");
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[0][0], rotation_matrix_[0][1], rotation_matrix_[0][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[1][0], rotation_matrix_[1][1], rotation_matrix_[1][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[2][0], rotation_matrix_[2][1], rotation_matrix_[2][2]);

  ::output.write("\ninverse rotation matrix mz -> m\n");
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[0][0], inverse_rotation_matrix_[0][1], inverse_rotation_matrix_[0][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[1][0], inverse_rotation_matrix_[1][1], inverse_rotation_matrix_[1][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[2][0], inverse_rotation_matrix_[2][1], inverse_rotation_matrix_[2][2]);


  // --- sanity check
  jblib::Vec3<double> test_unit_vec(0.0, 0.0, 1.0);
  jblib::Vec3<double> test_forward_vec = rotation_matrix_*test_unit_vec;
  jblib::Vec3<double> test_back_vec    = inverse_rotation_matrix_*test_forward_vec;

  ::output.write("\nrotation sanity check\n");

  ::output.write("  rotate      %f  %f  %f -> %f  %f  %f\n", test_unit_vec.x, test_unit_vec.y, test_unit_vec.z, test_forward_vec.x, test_forward_vec.y, test_forward_vec.z);
  ::output.write("  back rotate %f  %f  %f -> %f  %f  %f\n", test_forward_vec.x, test_forward_vec.y, test_forward_vec.z, test_back_vec.x, test_back_vec.y, test_back_vec.z);
  // ---

  outfile.open(std::string(::seedname + "_mc_stats.dat").c_str());
}

void ConstrainedMCSolver::calculate_trial_move(jblib::Vec3<double> &spin, const double move_sigma = 0.05) {
  jblib::Vec3<double> rvec;
  rng.sphere(rvec.x, rvec.y, rvec.z);
  spin += rvec*move_sigma;
  spin /= abs(spin);
}

void ConstrainedMCSolver::set_spin(const int &i, const jblib::Vec3<double> &spin) {
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    globals::s(i, n) = spin[n];
  }
}

void ConstrainedMCSolver::get_spin(const int &i, jblib::Vec3<double> &spin) {
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    spin[n] = globals::s(i, n);
  }
}

void ConstrainedMCSolver::run() {
  // Chooses nspins random spin pairs from the spin system and attempts a
  // Constrained Monte Carlo move on each pair, accepting for either lower
  // energy or with a Boltzmann thermal weighting.
  using namespace globals;

  std::string trial_step_name;
  // if (iteration_ % 2 == 0) {
  //   AsselinAlgorithm(mc_small_trial_step);
  //   trial_step_name = "STS";
  // } else {
    AsselinAlgorithm(mc_uniform_trial_step);
    trial_step_name = "UTS";
  // }
  // if (iteration_ % 2 == 0) {
  //   AsselinAlgorithm(mc_uniform_trial_step);
  //   trial_step_name = "UTS";
  // } else {
  //   if ((iteration_ - 1) % 4 == 0) {
  //     AsselinAlgorithm(mc_reflection_trial_step);
  //     trial_step_name = "RTS";
  //   } else {
  //     AsselinAlgorithm(mc_small_trial_step);
  //     trial_step_name = "STS";
  //   }
  // }

  move_acceptance_fraction_ = move_acceptance_count_/(0.5*num_spins);
  outfile << std::setw(8) << iteration_ << std::setw(8) << trial_step_name << std::fixed << std::setw(12) << move_acceptance_fraction_ << std::setw(12) << std::endl;

  iteration_++;
}

void ConstrainedMCSolver::AsselinAlgorithm(jblib::Vec3<double> (*mc_trial_step)(const jblib::Vec3<double>)) {
  using std::pow;
  using std::abs;
  using jblib::Vec3;

  int rand_s1, rand_s2;
  double delta_energy1, delta_energy2, delta_energy21;
  double mu1, mu2, mz_old, mz_new, probability;
  const double beta = kBohrMagneton/(physics_module_->temperature()*kBoltzmann);

  Vec3<double> s1_initial, s1_final, s1_initial_rotated, s1_final_rotated;
  Vec3<double> s2_initial, s2_final, s2_initial_rotated, s2_final_rotated;

  Vec3<double> m_other(0.0, 0.0, 0.0);

  for (int i = 0; i < globals::num_spins; ++i) {
    for (int n = 0; n < 3; ++n) {
      m_other[n] += globals::s(i,n)*globals::mus(i);
    }
  }

  if (abs(rad_to_deg(acos(m_other.z/abs(m_other))) - constraint_theta_) > 1e-5 ) {
    std::stringstream ss;
    ss << "ConstrainedMCSolver::AsselinAlgorithm -- theta constraint violated (" << rad_to_deg(acos(m_other.z/abs(m_other))) << " deg)";
    throw std::runtime_error(ss.str());
  }

  if (abs(rad_to_deg(atan2(m_other.y, m_other.x)) - constraint_phi_) > 1e-5 ) {
    std::stringstream ss;
    ss << "ConstrainedMCSolver::AsselinAlgorithm -- phi constraint violated (" << rad_to_deg(atan2(m_other.y, m_other.x)) << " deg)";
  }

  move_acceptance_count_ = 0;

  // In this 'for' loop I've tried to bail and continue at the earliest possible point
  // in all cases to try and avoid extra unneccesary calculations when the move will be
  // rejected.

  // we move two spins moving all spins on average is num_spins/2
  for (int i = 0; i < globals::num_spins/2; ++i) {

    // randomly get two spins s1 != s2
    rand_s1 = rng.uniform_discrete(0, globals::num_spins-1);
    rand_s2 = rand_s1;
    while (rand_s2 == rand_s1) {
      rand_s2 = rng.uniform_discrete(0, globals::num_spins-1);
    }

    s1_initial = mc_spin_as_vec(rand_s1);
    s2_initial = mc_spin_as_vec(rand_s2);
    mu1 = globals::mus(rand_s1);
    mu2 = globals::mus(rand_s2);

    // rotate into reference frame of the constraint vector
    s1_initial_rotated = rotation_matrix_*s1_initial;
    s2_initial_rotated = rotation_matrix_*s2_initial;

    // Monte Carlo move
    s1_final = s1_initial;
    s1_final = mc_trial_step(s1_initial);
    s1_final_rotated = rotation_matrix_*s1_final;

    // calculate new spin based on contraint mx = my = 0 in the constraint vector reference frame
    s2_final_rotated = (s1_initial_rotated - s1_final_rotated)*(mu1/mu2) + s2_initial_rotated;

    // zero out the z-component which will be calculated below
    s2_final_rotated.z = 0.0;

    if (unlikely(dot(s2_final_rotated, s2_final_rotated) > 1.0)) {
      // the rotated spin does not fit on the unit sphere - revert s1 and reject move
      continue;
    }
    // calculate the z-component so that |s2| = 1
    s2_final_rotated.z = copysign(1.0, s2_initial_rotated.z)*sqrt(1.0 - dot(s2_final_rotated, s2_final_rotated));

    // rotate s2 back into the cartesian reference frame
    s2_final = inverse_rotation_matrix_*s2_final_rotated;

    mz_new = dot((m_other + s1_final*mu1 + s2_final*mu2 - s1_initial*mu1 - s2_initial*mu2), constraint_vector_);

    // The new magnetization is in the opposite sense - revert s1, reject move
    if (unlikely(mz_new < 0.0)) {
      continue;
    }

    // change in energy with spin move
    delta_energy1 = 0.0;

    for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
      delta_energy1 += (*it)->calculate_one_spin_energy_difference(rand_s1, s1_initial, s1_final);
    }

    // temporarily accept the move for s1 so we can calculate the s2 energies
    // this will be reversed later if the move is rejected
    mc_set_spin_as_vec(rand_s1, s1_final);

    delta_energy2 = 0.0;
    for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
      delta_energy2 += (*it)->calculate_one_spin_energy_difference(rand_s2, s2_initial, s2_final);
    }

    // calculate the total energy difference
    delta_energy21 = delta_energy1 + delta_energy2;

    mz_old = dot(m_other, constraint_vector_);

    // calculate the Boltzmann weighted probability including the Jacobian factors (see paper)
    probability = exp(-delta_energy21*beta)*(pow(mz_new/mz_old, 2))*abs(s2_initial_rotated.z/s2_final_rotated.z);

    if (probability < rng.uniform()) {
      // move fails to overcome Boltzmann factor - revert s1, reject move
      mc_set_spin_as_vec(rand_s1, s1_initial);
      continue;
    } else {
      // accept move
      mc_set_spin_as_vec(rand_s2, s2_final);
      m_other += s1_final*mu1 + s2_final*mu2 - s1_initial*mu1 - s2_initial*mu2;
      move_acceptance_count_++;
    }
  }
}
