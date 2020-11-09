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

using namespace std;

std::vector<double> MetadynamicsMetropolisSolver::linear_space(const double &min,const double &max,const double &step) {
  assert(min < max);
  vector<double> space;
  double value = min;
  do {
	space.push_back(value);
	value += step;
  } while (value < max+step);

  return space;
}

void
MetadynamicsMetropolisSolver::initialize(const libconfig::Setting &settings) {
  MetropolisMCSolver::initialize(settings);
}

void MetadynamicsMetropolisSolver::run() {
  // put code here for inserting new gaussians

  MetropolisMCSolver::run();
}

double MetadynamicsMetropolisSolver::energy_difference(const int spin_index,
                                                       const Vec3 &initial_Spin,
                                                       const Vec3 &final_Spin) {

  return MetropolisMCSolver::energy_difference(spin_index, initial_Spin, final_Spin)
         + potential_difference(spin_index, initial_Spin, final_Spin);
}

double MetadynamicsMetropolisSolver::potential_difference(const int spin_index,
                                                          const Vec3 &initial_Spin,
                                                          const Vec3 &final_Spin) {
  return 0;
}
