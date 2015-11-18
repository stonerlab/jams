// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/metropolismc.h"

#include "core/maths.h"
#include "core/consts.h"
#include "core/globals.h"
#include "core/montecarlo.h"
#include "core/hamiltonian.h"

#include <iomanip>

MetropolisMCSolver::~MetropolisMCSolver() {
  if (outfile.is_open()) {
    outfile.close();
  }
}

void MetropolisMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

  // initialize base class
  Solver::initialize(argc, argv, idt);

  output.write("Initialising Metropolis Monte-Carlo solver\n");

  is_preconditioner_enabled_ = false;
  preconditioner_delta_theta_ = 5.0;
  preconditioner_delta_phi_ = 5.0;
  if (config.exists("sim.preconditioner")) {
    is_preconditioner_enabled_ = true;
    preconditioner_delta_theta_ = config.lookup("sim.preconditioner.[0]");
    preconditioner_delta_phi_ = config.lookup("sim.preconditioner.[1]");
  }

  outfile.open(std::string(::seedname + "_mc_stats.dat").c_str());
}

  void MetropolisMCSolver::run() {
    using namespace globals;

    if (is_preconditioner_enabled_ && iteration_ == 0) {
      output.write("Preconditioning...");

      // do a short thermalization
      for (int i = 0; i < 500; ++i) {
        MetropolisAlgorithm(mc_uniform_trial_step);
      }

      // now try systematic rotations
      SystematicPreconditioner(preconditioner_delta_theta_, preconditioner_delta_phi_);
      output.write("done\n");
    }

    std::string trial_step_name;
    if (iteration_ % 2 == 0) {
      MetropolisAlgorithm(mc_small_trial_step);
      trial_step_name = "STS";
    } else {
      MetropolisAlgorithm(mc_uniform_trial_step);
      trial_step_name = "UTS";
    }

    // if (iteration_ % 2 == 0) {
    //   MetropolisAlgorithm(mc_uniform_trial_step);
    // } else {
    //   if ((iteration_ - 1) % 4 == 0) {
    //     MetropolisAlgorithm(mc_reflection_trial_step);
    //   } else {
    //     MetropolisAlgorithm(mc_small_trial_step);
    //   }
    // }

    move_acceptance_fraction_ = move_acceptance_count_/double(num_spins);
    outfile << std::setw(8) << iteration_ << std::setw(8) << trial_step_name << std::fixed << std::setw(12) << move_acceptance_fraction_ << std::setw(12) << std::endl;

    iteration_++;
  }

  void MetropolisMCSolver::MetropolisPreconditioner(jblib::Vec3<double> (*mc_trial_step)(const jblib::Vec3<double>)) {
    int n;
    double e_initial, e_final;
    jblib::Vec3<double> s_initial, s_final;

    s_initial = mc_spin_as_vec(0);
    s_final = mc_trial_step(s_initial);

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

  void MetropolisMCSolver::SystematicPreconditioner(const double delta_theta, const double delta_phi) {
    // TODO: this should probably rotate spins rather than set definite direction so we can then handle
    // ferrimagnets too
    int n, m, num_theta, num_phi;
    double e_min, e_final, theta, phi;

    jblib::Vec3<double> s_new;

    jblib::Array<double, 2> s_init(globals::s);
    jblib::Array<double, 2> s_min(globals::s);

    double theta_min = 0.0;
    double phi_min = 0.0;

    num_theta = (180.0 / delta_theta) + 1;
    num_phi   = (360.0 / delta_phi);

    output.write("d_theta, d_phi (deg)\n");
    output.write("  %f, %f\n", delta_theta, delta_phi);

    output.write("num_theta, num_phi\n");
    output.write("  %d, %d\n", num_theta, num_phi);


    std::ofstream preconditioner_file;
    preconditioner_file.open(std::string(::seedname+"_mc_pre.dat").c_str());
    preconditioner_file << "# theta (deg) | phi (deg) | energy (J) \n";

    e_min = 1e10;
    theta = 0.0;
    for (int i = 0; i < num_theta; ++i) {
      phi = 0.0;
      for (int j = 0; j < num_phi; ++j) {

        const double c_t = cos(deg_to_rad(theta));
        const double c_p = cos(deg_to_rad(phi));
        const double s_t = sin(deg_to_rad(theta));
        const double s_p = sin(deg_to_rad(phi));

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

        jblib::Matrix<double, 3, 3> rotation_matrix = r_y*r_z;

        for (n = 0; n < globals::num_spins; ++n) {
          for (m = 0; m < 3; ++m) {
            s_new[m] = s_init(n, m);
          }
          s_new = rotation_matrix * s_new;
          for (m = 0; m < 3; ++m) {
            globals::s(n, m) = s_new[m];
          }
        }

        e_final = 0.0;
        for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
          e_final += (*it)->calculate_total_energy();
        }

        preconditioner_file << theta << "\t" << phi << "\t" << "\t" << e_final * kBohrMagneton << "\n";

        if ( e_final < e_min ) {
          // this configuration is the new minimum
          e_min = e_final;
          s_min = globals::s;
          theta_min = theta;
          phi_min = phi;
        }

        phi += delta_phi;
      }
      preconditioner_file << "\n\n";
      theta += delta_theta;
    }
    preconditioner_file.close();

    // use the minimum configuration
    globals::s = s_min;
  }

  void MetropolisMCSolver::MetropolisAlgorithm(jblib::Vec3<double> (*mc_trial_step)(const jblib::Vec3<double>)) {
    const double beta = kBohrMagneton/(kBoltzmann*physics_module_->temperature());
    int n, random_spin_number;
    double deltaE = 0.0;
    jblib::Vec3<double> s_initial, s_final;

    move_acceptance_count_ = 0;
    for (n = 0; n < globals::num_spins; ++n) {
      random_spin_number = rng.uniform_discrete(0, globals::num_spins - 1);
      s_initial = mc_spin_as_vec(random_spin_number);
      s_final = mc_trial_step(s_initial);

      deltaE = 0.0;
      for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
        deltaE += (*it)->calculate_one_spin_energy_difference(random_spin_number, s_initial, s_final);
      }

      if (deltaE < 0.0) {
        mc_set_spin_as_vec(random_spin_number, s_final);
        move_acceptance_count_++;
        continue;
      }

      if (rng.uniform() < exp(-deltaE*beta)) {
        move_acceptance_count_++;
        mc_set_spin_as_vec(random_spin_number, s_final);
      }
    }
  }