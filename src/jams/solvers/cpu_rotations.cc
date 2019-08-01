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


using namespace std;

void RotationSolver::initialize(const libconfig::Setting& settings) {
  using namespace globals;

  // initialize base class
  Solver::initialize(settings);

  num_theta_ = jams::config_optional<unsigned>(settings, "num_theta", num_theta_);
  num_phi_ = jams::config_optional<unsigned>(settings, "num_phi", num_phi_);

  initialized_ = true;
}

void RotationSolver::run() {
  using namespace globals;

  const double dtheta = kPi / double(num_theta_ - 1);
  const double dphi = kTwoPi / double(num_phi_ - 1);

  for (auto i = 0; i < lattice->num_motif_atoms(); ++i) {

    Vec3 spin_initial = {s(i,0), s(i,1), s(i,2)};

    std::ofstream tsv_file(seedname + "_" + to_string(i) + "_ang_eng.tsv");
    tsv_file.width(12);
    tsv_file << "theta_deg\t";
    tsv_file << "phi_deg\t";
    for (auto &hamiltonian : solver->hamiltonians()) {
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
          s(i,j) = spin[j];
        }

        // print angles and energy
        tsv_file << rad_to_deg(theta) << "\t" << rad_to_deg(phi) << "\t";
        for (auto &hamiltonian : solver->hamiltonians()) {
          auto energy = kBohrMagneton * hamiltonian->calculate_one_spin_energy(i);
          tsv_file << std::scientific << std::setprecision(15) << energy << "\t";
        }
        tsv_file << std::endl;

        phi += dphi;
      }
      theta += dtheta;
    }

    for (auto j = 0; j < 3; ++j) {
      s(i,j) = spin_initial[j];
    }

    tsv_file.close();
  }
  iteration_++;
}
