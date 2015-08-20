// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/metropolismc.h"

#include "core/maths.h"
#include "core/consts.h"
#include "core/globals.h"
#include "core/montecarlo.h"
#include "core/hamiltonian.h"

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
}

  void MetropolisMCSolver::run() {
    using namespace globals;

    if (is_preconditioner_enabled_ && iteration_ == 0) {
      output.write("Preconditioning...");
      SystematicPreconditioner(preconditioner_delta_theta_, preconditioner_delta_phi_);
      output.write("done\n");
    }

    if (iteration_ % 2 == 0) {
      MetropolisAlgorithm(mc_uniform_trial_step);
    } else {
      if ((iteration_ - 1) % 4 == 0) {
        MetropolisAlgorithm(mc_reflection_trial_step);
      } else {
        MetropolisAlgorithm(mc_small_trial_step);
      }
    }
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

    // std::cerr << e_initial << "\t" << e_final << std::endl;

    if (e_final - e_initial > 0.0) {
      for (n = 0; n < globals::num_spins; ++n) {
        mc_set_spin_as_vec(n, s_initial);
      }
    }
  }

  void MetropolisMCSolver::SystematicPreconditioner(const double delta_theta, const double delta_phi) {
    // TODO: this should probably rotate spins rather than set definite direction so we can then handle
    // ferrimagnets too
    int n, num_theta, num_phi;
    double e_initial, e_final, theta, phi;
    jblib::Vec3<double> s_min, s_new;

    s_min = jblib::Vec3<double>(0.0, 0.0, 1.0);

    num_theta = (180.0 / delta_theta) + 1;
    num_phi   = (360.0 / delta_phi) + 1;

    std::ofstream preconditioner_file;
    preconditioner_file.open(std::string(::seedname+"_mc_pre.dat").c_str());
    preconditioner_file << "# theta (deg) | phi (deg) | energy (J) \n";

    theta = 0.0;
    for (int i = 0; i < num_theta; ++i) {
      phi = 0.0;
      for (int j = 0; j < num_phi; ++j) {
        s_new.x = sin(deg_to_rad(theta))*cos(deg_to_rad(phi));
        s_new.y = sin(deg_to_rad(theta))*sin(deg_to_rad(phi));
        s_new.z = cos(deg_to_rad(theta));

        e_initial = 0.0;
        for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
          e_initial += (*it)->calculate_total_energy();
        }

        for (n = 0; n < globals::num_spins; ++n) {
          mc_set_spin_as_vec(n, s_new);
        }

        e_final = 0.0;
        for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
          e_final += (*it)->calculate_total_energy();
        }

        preconditioner_file << theta << "\t" << phi << "\t" << "\t" << e_final*kBohrMagneton << "\n";

        if ( ! (e_final < e_initial) ) {
          for (n = 0; n < globals::num_spins; ++n) {
            mc_set_spin_as_vec(n, s_min);
          }
        } else {
          s_min = s_new;
        }

        phi += delta_phi;
      }
      preconditioner_file << "\n\n";
      theta += delta_theta;
    }
    preconditioner_file.close();
  }

  void MetropolisMCSolver::MetropolisAlgorithm(jblib::Vec3<double> (*mc_trial_step)(const jblib::Vec3<double>)) {
    const double beta = kBohrMagneton/(kBoltzmann*physics_module_->temperature());
    int n, random_spin_number;
    double deltaE = 0.0;
    jblib::Vec3<double> s_initial, s_final;

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
        continue;
      }

      if (rng.uniform() < exp(-deltaE*beta)) {
        mc_set_spin_as_vec(random_spin_number, s_final);
      }
    }
  }