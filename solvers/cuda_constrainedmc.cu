// Copyright 2014 Joseph Barker. All rights reserved.
#include "core/cuda_solver_kernels.h"

#include "solvers/cuda_constrainedmc.h"
#include "core/cuda_sparsematrix.h"
#include "core/consts.h"
#include "core/maths.h"
#include "core/globals.h"

#include <iomanip>

void CudaConstrainedMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

    // initialize base class
  CudaSolver::initialize(argc, argv, idt);

  move_acceptance_count_ = 0;
  move_acceptance_fraction_ = 0.234;
  move_sigma_ = 0.001;

  ::output.write("Initialising Constrained Monte-Carlo solver\n");

  libconfig::Setting &solver_settings = ::config.lookup("sim");

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

  //if (verbose_output_is_set) {
    jblib::Vec3<double> test_unit_vec(0.0, 0.0, 1.0);
    jblib::Vec3<double> test_forward_vec = rotation_matrix_*test_unit_vec;
    jblib::Vec3<double> test_back_vec    = inverse_rotation_matrix_*test_forward_vec;

    ::output.write("\nsanity check\n");

    ::output.write("  rotate      %f  %f  %f -> %f  %f  %f\n", test_unit_vec.x, test_unit_vec.y, test_unit_vec.z, test_forward_vec.x, test_forward_vec.y, test_forward_vec.z);
    ::output.write("  back rotate %f  %f  %f -> %f  %f  %f\n", test_forward_vec.x, test_forward_vec.y, test_forward_vec.z, test_back_vec.x, test_back_vec.y, test_back_vec.z);
  //}

    std::string name = seedname + "_mc.dat";
    outfile.open(name.c_str());
    outfile.setf(std::ios::right);
    outfile << "#";
    outfile << std::setw(8) << "iteration";
    outfile << std::setw(12) << "acceptance";
    outfile << std::setw(12) << "sigma" << std::endl;
}


void CudaConstrainedMCSolver::calculate_trial_move(jblib::Vec3<double> &spin, const double move_sigma = 0.05) {
  double x,y,z;
  rng.sphere(x,y,z);
  spin.x += move_sigma*x; spin.y += move_sigma*y; spin.z += move_sigma*z;
  spin /= abs(spin);
}

double CudaConstrainedMCSolver::compute_one_spin_energy(const jblib::Vec3<double> &s_final, const int &ii) {
  using namespace globals;

  double energy_initial = 0.0;
  double energy_final = 0.0;
  double field[3];


  if (optimize::use_fft) {

    cuda_realspace_to_kspace_mapping<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(dev_s_.data(), r_to_k_mapping_.data(), num_spins, num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, dev_s3d_.data());

    if (cufftExecD2Z(spin_fft_forward_transform, dev_s3d_.data(), dev_sq_.data()) != CUFFT_SUCCESS) {
      jams_error("CUFFT failure executing spin_fft_forward_transform");
    }

    const int convolution_size = num_kpoints_.x*num_kpoints_.y*((num_kpoints_.z/2)+1);
    const int real_size = num_kpoints_.x*num_kpoints_.y*num_kpoints_.z;

    cuda_fft_convolution<<<(convolution_size+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE >>>(convolution_size, real_size, dev_wq_.data(), dev_sq_.data(), dev_hq_.data());
    if (cufftExecZ2D(field_fft_backward_transform, dev_hq_.data(), dev_h3d_.data()) != CUFFT_SUCCESS) {
      jams_error("CUFFT failure executing field_fft_backward_transform");
    }

    cuda_kspace_to_realspace_mapping<<<(num_spins+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(dev_h3d_.data(), r_to_k_mapping_.data(), num_spins, num_kpoints_.x, num_kpoints_.y, num_kpoints_.z, dev_h_.data());
  }

  if(J1ij_t.nonZero() > 0){
    spmv_dia_kernel<<< dev_J1ij_t_.blocks, DIA_BLOCK_SIZE >>>
    (num_spins3, num_spins3, J1ij_t.diags(), dev_J1ij_t_.pitch, 1.0, 1.0,
     dev_J1ij_t_.row, dev_J1ij_t_.val, dev_s_.data(), dev_h_.data());
  }

  // extract field for the site we care about
  cudaMemcpy(&field[0], (dev_h_.data()+3*ii), 3*sizeof(double), cudaMemcpyDeviceToHost);

  // dipole and exchange
  energy_initial -= s(ii, 0)*field[0] + s(ii, 1)*field[1] + s(ii,2)*field[2];
  // anisotropy
  energy_initial -= d2z(ii)*0.5*(3.0*s(ii, 2)*s(ii, 2) - 1.0);
  energy_initial -= d4z(ii)*0.125*(35.0*s(ii, 2)*s(ii, 2)*s(ii, 2)*s(ii, 2)-30.0*s(ii, 2)*s(ii, 2) + 3.0);
  energy_initial -= d6z(ii)*0.0625*(231.0*s(ii, 2)*s(ii, 2)*s(ii, 2)*s(ii, 2)*s(ii, 2)*s(ii, 2) - 315.0*s(ii, 2)*s(ii, 2)*s(ii, 2)*s(ii, 2) + 105.0*s(ii, 2)*s(ii, 2) - 5.0);

  // dipole and exchange
  energy_final -= s_final[0]*field[0] + s_final[1]*field[1] + s_final[2]*field[2];
  // anisotropy
  energy_final -= d2z(ii)*0.5*(3.0*s_final[2]*s_final[2] - 1.0);
  energy_final -= d4z(ii)*0.125*(35.0*s_final[2]*s_final[2]*s_final[2]*s_final[2]-30.0*s_final[2]*s_final[2] + 3.0);
  energy_final -= d6z(ii)*0.0625*(231.0*s_final[2]*s_final[2]*s_final[2]*s_final[2]*s_final[2]*s_final[2] - 315.0*s_final[2]*s_final[2]*s_final[2]*s_final[2] + 105.0*s_final[2]*s_final[2] - 5.0);

  return (energy_final - energy_initial);
}

  void CudaConstrainedMCSolver::run() {
    // Chooses nspins random spin pairs from the spin system and attempts a
    // Constrained Monte Carlo move on each pair, accepting for either lower
    // energy or with a Boltzmann thermal weighting.
    //
    // NOTES:
    // The the Random class (global name rng) has a method uniform_discrete(min,max)
    // which does a truely uniform sampling in the integer range [min, max]. Doing
    // rng.uniform()*num_spins can have a small bias depending on the value of num_spins.\
    //
    using namespace globals;

    const double inv_kbT_bohr = mu_bohr_si/(physics_module_->temperature()*boltzmann_si);

    // assuming an optimal acceptance rate of 0.234 [1 A. Gelman, G. Roberts, and W. Gilks, Bayesian Statistics 5, 599 (1996)]
    // try to keep within 0.100 of this value for any given sample run through the system
    // if (iteration_%50 == 0) {
    //     if (move_acceptance_fraction_ < 0.15) {
    //       move_sigma_ = 0.5*move_sigma_;
    //       if (verbose_output_is_set) {
    //         ::output.write("CMC acceptance < 0.45 (%f), new sigma %f", move_acceptance_fraction_, move_sigma_);
    //       }
    //     } else if (move_acceptance_fraction_ > 0.40) {
    //       move_sigma_ = 2.0*move_sigma_;
    //       if (verbose_output_is_set) {
    //         ::output.write("CMC acceptance > 0.60 (%f), new sigma %f", move_acceptance_fraction_, move_sigma_);
    //       }
    //     }
    //     if (move_sigma_ < 0.00001) {
    //       move_sigma_ = 0.00001;
    //       if (verbose_output_is_set) {
    //         ::output.write("CMC sigma lower limit hit %f", move_sigma_);
    //       }
    //     }

    //     if (move_sigma_ > 2.0) {
    //       move_sigma_ = 2.0;
    //       if (verbose_output_is_set) {
    //         ::output.write("CMC sigma upper limit hit %f", move_sigma_);
    //       }
    //     }
    //     move_acceptance_count_ = 0;
    // }



    // calculate the initial magnetization before we try to move every spin
    jblib::Vec3<double> m_other(0.0, 0.0, 0.0);
    for (int i = 0; i < num_spins; ++i) {
      for (int n = 0; n < 3; ++n) {
        m_other[n] += s(i, n);
      }
    }

    // loop over every spin (on average), once but divide by 2 because we move
    // two spins each time
    move_acceptance_count_ = 0;
    for (int i = 0; i < num_spins/2; ++i) {
      // randomly select spin 1
      int rand_s1 = rng.uniform_discrete(0, num_spins-1);
      jblib::Vec3<double> s1_initial(s(rand_s1, 0), s(rand_s1, 1), s(rand_s1,2));

      // rotate into reference frame of the constraint vector
      jblib::Vec3<double> s1_initial_rotated = rotation_matrix_*s1_initial;

      // monte carlo move
      jblib::Vec3<double> s1_final = s1_initial;
      calculate_trial_move(s1_final);
      jblib::Vec3<double> s1_final_rotated = rotation_matrix_*s1_final;

      // change in energy with spin move
      const double delta_energy1 = compute_one_spin_energy(s1_final, rand_s1);

      // randomly select spin 2 for i != j
      int rand_s2 = rand_s1;
      while (rand_s2 == rand_s1) {
        rand_s2 = rng.uniform_discrete(0, num_spins-1);
      }
      jblib::Vec3<double> s2_initial(s(rand_s2, 0), s(rand_s2, 1), s(rand_s2, 2));
      jblib::Vec3<double> s2_initial_rotated = rotation_matrix_*s2_initial;

      // calculate new spin based on contraint mx = my = 0 in the constraint vector reference frame
      jblib::Vec3<double> s2_final_rotated(
        s1_initial_rotated.x + s2_initial_rotated.x - s1_final_rotated.x,
        s1_initial_rotated.y + s2_initial_rotated.y - s1_final_rotated.y,
        0.0);

      // check the rotated spin fits in the unit sphere, if not we will reject the move
      if (((s2_final_rotated.x*s2_final_rotated.x) + (s2_final_rotated.y*s2_final_rotated.y)) < 1.0) {
        // calculate the z-component so that |s2| = 1
        s2_final_rotated.z = sign(1.0, s2_initial_rotated.z)*sqrt(1.0 - (s2_final_rotated.x*s2_final_rotated.x) - (s2_final_rotated.y*s2_final_rotated.y));

        // rotate s2 back into the cartesian reference frame
        jblib::Vec3<double> s2_final = inverse_rotation_matrix_*s2_final_rotated;

        // temporarily accept the move for s1 so we can calculate the s2 energies
        // this will be reversed later if the move is rejected
        for (int n = 0; n < 3; ++n) {
          s(rand_s1, n) = s1_final[n];
        }
        // also update s1 on the CUDA device
        cudaMemcpy((dev_s_.data()+3*rand_s1), &s1_final[0], 3*sizeof(double), cudaMemcpyHostToDevice);

        // calculate the energy difference for s2
        const double delta_energy2 = compute_one_spin_energy(s2_final, rand_s2);

        // calculate the total energy difference
        const double delta_energy21 = delta_energy1+delta_energy2;

        double mz_old = dot(m_other, constraint_vector_);

        double mz_new = (m_other.x + s1_final.x + s2_final.x - s1_initial.x - s2_initial.x)*constraint_vector_.x
                       +(m_other.y + s1_final.y + s2_final.y - s1_initial.y - s2_initial.y)*constraint_vector_.y
                       +(m_other.z + s1_final.z + s2_final.z - s1_initial.z - s2_initial.z)*constraint_vector_.z;

        // calculate the Boltzmann weighted probability including the
        // Jacobian factors (see paper)
        const double probability = exp(-delta_energy21*inv_kbT_bohr)*((mz_new/mz_old)*(mz_new/mz_old))*fabs(s2_initial_rotated.z/s2_final_rotated.z);

        if (delta_energy21 < 0.0 && (mz_new >= 0.0)) {
            // moves reduce total energy -> accept
            for (int n = 0; n < 3; ++n) {  // accept s2
                s(rand_s2, n) = s2_final[n];
            }
            // update s2 on the CUDA device
            cudaMemcpy((dev_s_.data()+3*rand_s2), &s2_final[0], 3*sizeof(double), cudaMemcpyHostToDevice);
            move_acceptance_count_++;
        }else if (probability >= rng.uniform() && (mz_new >= 0.0)) {
            // moves overcome Boltzmann weighting -> accept
          for (int n = 0; n < 3; ++n) {  // accept s2
            s(rand_s2, n) = s2_final[n];
          }
          // update s2 on the CUDA device
          cudaMemcpy((dev_s_.data()+3*rand_s2), &s2_final[0], 3*sizeof(double), cudaMemcpyHostToDevice);
            move_acceptance_count_++;
        } else {
          for (int n = 0; n < 3; ++n) {  // revert s1
            s(rand_s1, n) = s1_initial[n];
          }
          // revert s1 on the CUDA device
          cudaMemcpy((dev_s_.data()+3*rand_s1), &s1_initial[0], 3*sizeof(double), cudaMemcpyHostToDevice);
        }
      } else {   // if s2 not on unit sphere
        for (int n = 0; n < 3; ++n) {
          s(rand_s1, n) = s1_initial[n];
        }
        // revert s1 on the CUDA device
        cudaMemcpy((dev_s_.data()+3*rand_s1), &s1_initial[0], 3*sizeof(double), cudaMemcpyHostToDevice);
      }
    }



    // compute fields ready for torque monitor
    compute_fields();
    dev_h_.copy_to_host_array(globals::h);

    move_acceptance_fraction_ = move_acceptance_count_/(0.5*num_spins);
    outfile << std::setw(8) << iteration_ << std::setw(12) << move_acceptance_fraction_ << std::setw(12) << move_sigma_ << std::endl;

    iteration_++;
  }

  void CudaConstrainedMCSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s) {
  }
