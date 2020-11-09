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
