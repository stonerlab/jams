//
// Created by Joseph Barker on 2019-05-03.
//

#include <cmath>
#include <fstream>

#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/solvers/cpu_rotations.h"

#include "jams/interface/config.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/helpers/output.h"

void RotationSolver::initialize(const libconfig::Setting& settings) {
  rotate_all_spins_= jams::config_optional<bool>(settings, "rotate_all_spins", rotate_all_spins_);
  num_theta_ = jams::config_optional<unsigned>(settings, "num_theta", num_theta_);
  num_phi_ = jams::config_optional<unsigned>(settings, "num_phi", num_phi_);
}

void RotationSolver::run() {
  const double dtheta = kPi / double(num_theta_ - 1);
  const double dphi = kTwoPi / double(num_phi_ - 1);

  if (rotate_all_spins_) {
    std::ofstream tsv_file(jams::output::full_path_filename("ang_eng.tsv"));
    tsv_file.width(12);
    tsv_file << "phi_deg\t";
    tsv_file << "theta_deg\t";
    for (auto &hamiltonian: this->hamiltonians()) {
      tsv_file << hamiltonian->name() << "_e\t";
    }
    tsv_file << std::endl;

    auto initial_spins = globals::s;

    double theta = 0.0;
    for (auto m = 0; m < num_theta_; ++m) {
      double phi = 0.0;
      for (auto n = 0; n < num_phi_; ++n) {

        globals::s = initial_spins;
        Mat3 rotation = rotation_matrix_z(-phi)*rotation_matrix_y(-theta);

        for (auto i = 0; i < globals::num_spins; ++i) {
          Vec3 spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
          spin = rotation * spin;
          for (auto j = 0; j < 3; ++j) {
            globals::s(i, j) = spin[j];
          }
        }

        // print angles and energy
        tsv_file << rad_to_deg(phi) << "\t" << rad_to_deg(theta) << "\t";
        for (auto &hamiltonian: this->hamiltonians()) {
          auto energy = hamiltonian->calculate_total_energy(this->time());
          tsv_file << std::scientific << std::setprecision(15) << energy
                   << "\t";
        }
        tsv_file << std::endl;

        phi += dphi;
      }
      theta += dtheta;
    }
    tsv_file.close();
  } else {

    for (auto i = 0; i < globals::lattice->num_basis_sites(); ++i) {

      Vec3 spin_initial = {globals::s(i, 0), globals::s(i, 1),
                           globals::s(i, 2)};

      std::ofstream tsv_file(
          jams::output::full_path_filename_series("ang_eng.tsv", i, 1));
      tsv_file.width(12);
      tsv_file << "phi_deg\t";
      tsv_file << "theta_deg\t";
      for (auto &hamiltonian: this->hamiltonians()) {
        tsv_file << hamiltonian->name() << "_e\t";
      }
      tsv_file << std::endl;

      double theta = 0.0;
      for (auto m = 0; m < num_theta_; ++m) {
        double phi = 0.0;
        for (auto n = 0; n < num_phi_; ++n) {

          // rotate one spin
          Vec3 spin = spherical_to_cartesian_vector(1.0, theta, phi);
          for (auto j = 0; j < 3; ++j) {
            globals::s(i, j) = spin[j];
          }

          // print angles and energy
          tsv_file << rad_to_deg(phi) << "\t" << rad_to_deg(theta) << "\t";
          for (auto &hamiltonian: this->hamiltonians()) {
            auto energy = hamiltonian->calculate_energy(i, this->time());
            tsv_file << std::scientific << std::setprecision(15) << energy
                     << "\t";
          }
          tsv_file << std::endl;

          phi += dphi;
        }
        theta += dtheta;
      }

      for (auto j = 0; j < 3; ++j) {
        globals::s(i, j) = spin_initial[j];
      }

      tsv_file.close();
    }
  }
  iteration_++;
  time_ = 0.0;
}
