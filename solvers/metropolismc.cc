// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/metropolismc.h"

#include "core/maths.h"
#include "core/consts.h"
#include "core/globals.h"
#include "core/montecarlo.h"
#include "core/hamiltonian.h"
#include "core/permutations.h"

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

  if (output.is_verbose()) {
    outfile.open(std::string(::seedname + "_mc_stats.dat").c_str());
  }
}

  void MetropolisMCSolver::run() {
    using namespace globals;

    if (is_preconditioner_enabled_ && iteration_ == 0) {
      output.write("preconditioning\n");

      output.write("  thermalizing\n");
      // do a short thermalization
      for (int i = 0; i < 500; ++i) {
        MetropolisAlgorithm(mc_uniform_trial_step);
      }

      // now try systematic rotations
      output.write("  magnetization rotations\n");
      SystematicPreconditioner(preconditioner_delta_theta_, preconditioner_delta_phi_);
      output.write("done\n");
    }

    std::string trial_step_name;

    if ((iteration_ + 1) % 4 == 0) {
      MetropolisAlgorithm(mc_small_trial_step);
      trial_step_name = "STS";
    } else if ((iteration_ + 1) % 5 == 0) {
      MetropolisAlgorithm(mc_reflection_trial_step);
      trial_step_name = "RTS";
    } else {
      MetropolisAlgorithm(mc_uniform_trial_step);
      trial_step_name = "UTS";
    }
      
    if (output.is_verbose()) {
      move_acceptance_fraction_ = move_acceptance_count_/double(num_spins);
      outfile << std::setw(8) << iteration_ << std::setw(8) << trial_step_name << std::fixed << std::setw(12) << move_acceptance_fraction_ << std::setw(12) << std::endl;
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

    if (e_final - e_initial > 0.0) {
      for (n = 0; n < globals::num_spins; ++n) {
        mc_set_spin_as_vec(n, s_initial);
      }
    }
  }

  class MetropolisMCSolver::MagnetizationRotationMinimizer
  {
      std::vector<Hamiltonian*> * hamiltonians_;

      std::uint64_t count;
      double e_min;
      jblib::Array<double, 2> s_min;
  public:
      explicit MagnetizationRotationMinimizer(std::vector<Hamiltonian*> & hamiltonians_ ) :
        hamiltonians_(&hamiltonians_), count(0), e_min(1e10), s_min(globals::s) {}

      jblib::Array<double, 2> s() {
        return s_min;
      }

      template <class It>
          bool operator()(It first, It last)  // called for each permutation
          {
            using std::vector;
            using jblib::Vec3;
            using jblib::Array;
            using jblib::Matrix;

            int i, j;
            double energy;
            Vec3<double> s_new;
            vector<Matrix<double, 3, 3>> rotation(lattice.num_materials());
            vector<Vec3<double>> mag(lattice.num_materials());

            if (last - first != lattice.num_materials()) {
              throw std::runtime_error("number of angles in preconditioner does not match the number of materials");
            }

            // count the number of times this is called
            ++count;

            // calculate magnetization vector of each material
            for (i = 0; i < globals::num_spins; ++i) {
              for (j = 0; j < 3; ++j) {
                mag[lattice.atom_material(i)][j] += globals::s(i, j);
              }
            }
            // don't need to normalize magnetization because only the direction is important
            // calculate rotation matrix between magnetization and desired direction
            for (i = 0; i < lattice.num_materials(); ++i) {
              rotation[i] = rotation_matrix_between_vectors(mag[i], spherical_to_cartesian_vector(1.0, *first, 0.0));
              ++first;
            }

            for (i = 0; i < globals::num_spins; ++i) {
              for (j = 0; j < 3; ++j) {
                s_new[j] = globals::s(i, j);
              }
              s_new = rotation[lattice.atom_material(i)] * s_new;
              for (j = 0; j < 3; ++j) {
                globals::s(i, j) = s_new[j];
              }
            }

            energy = 0.0;
            for (auto it = hamiltonians_->begin() ; it != hamiltonians_->end(); ++it) {
              energy += (*it)->calculate_total_energy();
            }

            if ( energy < e_min ) {
              // this configuration is the new minimum
              e_min = energy;
              s_min = globals::s;
            }
            return false;
          }

      operator std::uint64_t() const {return count;}
  };

  void MetropolisMCSolver::SystematicPreconditioner(const double delta_theta, const double delta_phi) {
    // TODO: this should probably rotate spins rather than set definite direction so we can then handle
    // ferrimagnets too
    int num_theta;
    // double e_min, e_final, phi;

    jblib::Vec3<double> s_new;

    jblib::Array<double, 2> s_init(globals::s);
    jblib::Array<double, 2> s_min(globals::s);

    num_theta = (180.0 / delta_theta) + 1;


    std::vector<double> theta(num_theta);

    theta[0] = 0.0;
    for (int i = 1; i < num_theta; ++i) {
      theta[i] = theta[i-1] + delta_theta;
    }

    MagnetizationRotationMinimizer minimizer(hamiltonians_);

    output.write("    delta theta (deg)\n");
    output.write("      %f\n", delta_theta);

    output.write("    num_theta\n");
    output.write("      %d\n", num_theta);

    std::uint64_t count = for_each_permutation(theta.begin(),
                                                   theta.begin() + 3,
                                                   theta.end(),
                                                   minimizer);

    output.write("    permutations\n");
    output.write("      %d\n", count);

    std::ofstream preconditioner_file;
    preconditioner_file.open(std::string(::seedname+"_mc_pre.dat").c_str());
    preconditioner_file << "# theta (deg) | phi (deg) | energy (J) \n";

    preconditioner_file.close();

    // use the minimum configuration
    globals::s = minimizer.s();
  }

  void MetropolisMCSolver::MetropolisAlgorithm(jblib::Vec3<double> (*mc_trial_step)(const jblib::Vec3<double>)) {
    const double beta = kBohrMagneton/(kBoltzmann*physics_module_->temperature());
    int n, random_spin_number;
    double deltaE = 0.0;
    jblib::Vec3<double> s_initial, s_final;

    move_acceptance_count_ = 0;
    for (n = 0; n < globals::num_spins; ++n) {

      // 2015-12-10 (JB) striding uniformly is ~4x faster than random choice (clang OSX).
      // Seems to be because of caching/predication in the exchange field calculation.
      random_spin_number = n; //rng.uniform_discrete(0, globals::num_spins - 1);

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
