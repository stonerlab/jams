//
// Created by ioannis charalampidis on 07/11/2020.
//

#include "cpu_metadynamics_metropolis_solver.h"
#include "jams/core/solver.h"

#include <fstream>
#include <jams/core/types.h>
#include <pcg_random.hpp>
#include <random>
#include "jams/helpers/random.h"
#include <vector>
#include <jams/core/globals.h>
#include <jams/helpers/montecarlo.h>
#include <iostream>
#include "jams/helpers/output.h"
#include <jams/maths/interpolation.h>
#include <jams/metadynamics/collective_variable_factory.h>

using namespace std;

void MetadynamicsMetropolisSolver::initialize(const libconfig::Setting &settings) {
  MetropolisMCSolver::initialize(settings);
  cv_potential_.reset(jams::CollectiveVariableFactory::create(settings));
}

void MetadynamicsMetropolisSolver::run() {
  // update the total magnetisation to avoid the possibility of drift with accumulated errors
  MetropolisMCSolver::run();
  if (iteration_ % 100 == 0) {
    cv_potential_->insert_gaussian(1.0);
  }
  if (iteration_ % 10000 == 0){
    ofstream potential_file(jams::output::full_path_filename("_cv_potential.tsv"));
    cv_potential_->output(potential_file);
    potential_file.close();
  }
}

double MetadynamicsMetropolisSolver::energy_difference(const int spin_index,
                                                       const Vec3 &initial_Spin,
                                                       const Vec3 &final_Spin) {

  return MetropolisMCSolver::energy_difference(spin_index, initial_Spin, final_Spin)
         + cv_potential_->potential_difference(spin_index, initial_Spin, final_Spin);
}

void MetadynamicsMetropolisSolver::accept_move(const int spin_index,
                                               const Vec3 &initial_spin,
                                               const Vec3 &final_spin) {
  MetropolisMCSolver::accept_move(spin_index, initial_spin, final_spin);
  cv_potential_->spin_update(spin_index, initial_spin, final_spin);
}

