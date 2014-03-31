// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/cuda_constrainedmc.h"

#include "core/consts.h"
#include "core/maths.h"
#include "core/globals.h"

void CudaConstrainedMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

    // initialize base class
  CudaSolver::initialize(argc, argv, idt);

  output.write("Initialising Constrained Monte-Carlo solver\n");



  libconfig::Setting &solver_settings = ::config.lookup("sim");

  constraint_theta_ = solver_settings["cmc_constraint_theta"];
  constraint_phi_   = solver_settings["cmc_constraint_phi"];

  ::output.write("\nconstraint angle theta (deg): % 8.8f\n", constraint_theta_);
  ::output.write("\nconstraint angle phi (deg): % 8.8f\n", constraint_phi_);

  constraint_vector_.x = cos(deg_to_rad(constraint_theta_))*sin(deg_to_rad(constraint_phi_));
  constraint_vector_.y = sin(deg_to_rad(constraint_theta_))*sin(deg_to_rad(constraint_phi_));
  constraint_vector_.z = cos(deg_to_rad(constraint_phi_));

  ::output.write("\nconstraint vector: % 8.8f, % 8.8f, % 8.8f\n", constraint_vector_.x, constraint_vector_.y, constraint_vector_.z);

  for (int i = 0; i < num_spins; ++i) {
    for (int n = 0; n < 3; ++ n) {
      s(i, n) = constraint_vector_[n];
    }
  }

  // calculate rotation matrix for rotating m -> mz
  const double c_t = cos(deg_to_rad(constraint_theta_));
  const double c_p = cos(deg_to_rad(constraint_phi_));
  const double s_t = sin(deg_to_rad(constraint_theta_));
  const double s_p = sin(deg_to_rad(constraint_phi_));

  jblib::Matrix<double, 3, 3> r_y;
  jblib::Matrix<double, 3, 3> r_z;

  r_y[0][0] =  c_p;  r_y[0][1] =  0.0;  r_y[0][2] =  s_p;
  r_y[1][0] =  0.0;  r_y[1][1] =  1.0;  r_y[1][2] =  0.0;
  r_y[2][0] = -s_p;  r_y[2][1] =  0.0;  r_y[2][2] =  c_p;

  r_z[0][0] =  c_t;  r_z[0][1] = -s_t;  r_z[0][2] =  0.0;
  r_z[1][0] =  s_t;  r_z[1][1] =  c_t;  r_z[1][2] =  0.0;
  r_z[2][0] =  0.0;  r_z[2][1] =  0.0;  r_z[2][2] =  1.0;

  rotation_matrix_ = r_y*r_z;

  ::output.write("\nrotation matrix m -> mz\n");
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[0][0], rotation_matrix_[0][1], rotation_matrix_[0][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[1][0], rotation_matrix_[1][1], rotation_matrix_[1][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", rotation_matrix_[2][0], rotation_matrix_[2][1], rotation_matrix_[2][2]);

  inverse_rotation_matrix_ = rotation_matrix_.transpose();
  ::output.write("\ninverse rotation matrix mz -> m\n");
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[0][0], inverse_rotation_matrix_[0][1], inverse_rotation_matrix_[0][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[1][0], inverse_rotation_matrix_[1][1], inverse_rotation_matrix_[1][2]);
  ::output.write("  % 8.8f  % 8.8f  % 8.8f\n", inverse_rotation_matrix_[2][0], inverse_rotation_matrix_[2][1], inverse_rotation_matrix_[2][2]);


  // output.write("\nconverting symmetric to general MAP matrices\n");

  // J1ij_t.convertSymmetric2General();

  // output.write("\nconverting MAP to CSR\n");

  // J1ij_t.convertMAP2CSR();

  // output.write("\nJ1ij Tensor matrix memory (CSR): %f MB\n",

  // J1ij_t.calculateMemory());
}


void CudaConstrainedMCSolver::calculate_trial_move(jblib::Vec3<double> &spin) {
  double x,y,z;
  rng.sphere(x,y,z);
  spin.x += 0.2*x;
  spin.y += 0.2*y;
  spin.z += 0.2*z;
  spin /= abs(spin);
}

double CudaConstrainedMCSolver::compute_one_spin_energy(const jblib::Vec3<double> &s_final, const int &ii) {
  using namespace globals;

  double energy_initial = 0.0;
  double energy_final = 0.0;

  compute_fields();

  double field[3];

  // extract field for the site we care about
  cudaMemcpy(&field[0], (dev_h_.data()+3*ii), 3*sizeof(double), cudaMemcpyDeviceToHost);

  energy_initial -= s(ii, 0)*field[0] + s(ii, 1)*field[1] + s(ii,2)*field[2];

  energy_final -= s_final[0]*field[0] + s_final[1]*field[1] + s_final[2]*field[2];

  return (energy_final - energy_initial);
}

  void CudaConstrainedMCSolver::run() {
    // Chooses nspins random spin pairs from the spin system and attempts a
    // Constrained Monte Carlo move on each pair, accepting for either lower
    // energy or with a Boltzmann thermal weighting.
    using namespace globals;

    const double inv_kbT_bohr = mu_bohr_si/(physics_module_->temperature()*boltzmann_si);

    jblib::Vec3<double> m_other(0.0, 0.0, 0.0);
    for (int i = 0; i < num_spins; ++i) {
      for (int n = 0; n < 3; ++n) {
        m_other[n] += s(i,n);
      }
    }

    std::cout << iteration_ << std::endl;

    for (int i = 0; i < num_spins/2; ++i) {
      // std::cout << i << std::endl;
      int rand_s1 = rng.uniform_discrete(0, num_spins-1);

      jblib::Vec3<double> s1_initial(s(rand_s1, 0), s(rand_s1, 1), s(rand_s1,2));

      jblib::Vec3<double> s1_initial_rotated = rotation_matrix_*s1_initial;


      // monte carlo move
      jblib::Vec3<double> s1_final = s1_initial;
      calculate_trial_move(s1_final);

      jblib::Vec3<double> s1_final_rotated = rotation_matrix_*s1_final;

      // CALCULATE DELTA E
      const double delta_energy1 = compute_one_spin_energy(s1_final, rand_s1);

      // randomly select spin number 2 (i/=j)
      int rand_s2 = rand_s1;
      while (rand_s2 == rand_s1) {
        rand_s2 = rng.uniform_discrete(0, num_spins-1);
      }

      jblib::Vec3<double> s2_initial(s(rand_s2, 0), s(rand_s2, 1), s(rand_s2, 2));

      jblib::Vec3<double> s2_initial_rotated = rotation_matrix_*s2_initial;

      // calculate new spin based on contraint mx=my=0
      jblib::Vec3<double> s2_final_rotated(s1_initial_rotated.x + s2_initial_rotated.x - s1_final_rotated.x,
                                           s1_initial_rotated.y + s2_initial_rotated.y - s1_final_rotated.y,
                                           0.0);

      if (((s2_final_rotated.x*s2_final_rotated.x) + (s2_final_rotated.y*s2_final_rotated.y)) < 1.0) {
        s2_final_rotated.z = sign(1.0, s2_initial_rotated.z)*sqrt(1.0 - (s2_final_rotated.x*s2_final_rotated.x) - (s2_final_rotated.y*s2_final_rotated.y));

        jblib::Vec3<double> s2_final = inverse_rotation_matrix_*s2_final_rotated;

        // automatically accept the move for spin1
        for (int n = 0; n < 3; ++n) {
          s(rand_s1, n) = s1_final[n];
        }

        // also update the CUDA device
        cudaMemcpy((dev_s_.data()+3*rand_s1), &s1_final[0], 3*sizeof(double), cudaMemcpyHostToDevice);

        // CALCULATE THE DETLA E FOR S2
        const double delta_energy2 = compute_one_spin_energy(s2_final, rand_s2);

        // CALCULATE THE DELTA E FOR BOTH SPINS
        const double delta_energy21 = delta_energy1+delta_energy2;

        double mz_old = dot(m_other, constraint_vector_);

        double mz_new = (m_other.x + s1_final.x + s2_final.x - s1_initial.x - s2_initial.x)*constraint_vector_.x
                       +(m_other.y + s1_final.y + s2_final.y - s1_initial.y - s2_initial.y)*constraint_vector_.y
                       +(m_other.z + s1_final.z + s2_final.z - s1_initial.z - s2_initial.z)*constraint_vector_.z;

        const double probability = exp(-delta_energy21*inv_kbT_bohr)*((mz_new/mz_old)*(mz_new/mz_old))*fabs(s2_initial_rotated.z/s2_final_rotated.z);

        if (probability >= rng.uniform() && (mz_new >= 0.0)) {
          for (int n = 0; n < 3; ++n) {  // accept s2
            s(rand_s2, n) = s2_final[n];
          }
          // also update the CUDA device
          cudaMemcpy((dev_s_.data()+3*rand_s2), &s2_final[0], 3*sizeof(double), cudaMemcpyHostToDevice);
        } else {
          for (int n = 0; n < 3; ++n) {  // revert s1
            s(rand_s1, n) = s1_initial[n];
          }
          // also revert the CUDA device
          cudaMemcpy((dev_s_.data()+3*rand_s1), &s1_initial[0], 3*sizeof(double), cudaMemcpyHostToDevice);
        }
      } else {   // if s2 not on unit sphere
      //   for (int n = 0; n < 3; ++n) {
      //     s(rand_s1, n) = s1_initial[n];
      //   }
      //   for (int n = 0; n < 3; ++n) {
      //     s(rand_s2, n) = s2_initial[n];
      //   }
      }
    }

    // compute fields ready for torque monitor
    compute_fields();
    dev_h_.copy_to_host_array(globals::h);
    iteration_++;
  }

  void CudaConstrainedMCSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s) {
  }
