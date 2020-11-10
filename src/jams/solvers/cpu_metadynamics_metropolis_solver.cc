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

void
MetadynamicsMetropolisSolver::initialize(const libconfig::Setting &settings) {
  MetropolisMCSolver::initialize(settings);
}

void MetadynamicsMetropolisSolver::run() {
  intialise_potential_histograms(potential_1D,potential_2D,sample_points_1d,sample_points_2d,sample_points_m_perpendicular);
  //to insert gaussian we need to know the magnetisation value at any given point of the simulation, perform the interpolation and use the function.
  //Also need to be able to just calculate the potential difference between 2 given points.
  // should we use flags to specify when we need to calculate the energy difference and when to add a gausian to our histogram ?!

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

double MetadynamicsMetropolisSolver::gaussian(const double &x,const double &center,const double &amplitude,const double &width) {
  return amplitude*exp(- (x-center)*(x-center ) / (2.0 * width*width));
}

void MetadynamicsMetropolisSolver::intialise_potential_histograms(std::vector<double> &potential_1D, std::vector<std::vector<double>> &potential_2D,const std::vector<double> sample_points_1d,const std::vector<double> sample_points_2d, const std::vector<double> sample_points_m_perpendicular)  {

  potential_1D.resize(sample_points_1d.size(),0.0);
  potential_2D.resize(sample_points_2d.size(),sample_points_m_perpendicular);
}

void MetadynamicsMetropolisSolver::insert_gaussian(const double &center, const double &amplitude, const double &width, const vector<double> &sample_points, vector<double> &discrete_potential) {
  assert(sample_points.size() == discrete_potential.size());
  for (auto i = 0; i < sample_points.size(); ++i) {
    discrete_potential[i] += gaussian(sample_points[i], center, amplitude, width);
  }
  // calculate the center position for a gaussian according to mirror boundary conditions
  double mirrored_center;
  if (center >=0) {
    mirrored_center = 2 - center;
  } else {
    mirrored_center = -2 - center;
  }
  assert(mirrored_center >= -2 && mirrored_center <= 2);

  // insert the mirrored gaussian
  for (auto i = 0; i < sample_points.size(); ++i) {
    discrete_potential[i] += gaussian(sample_points[i], mirrored_center, amplitude, width);
  }
}
double MetadynamicsMetropolisSolver::interpolated_potential(const vector<double> &sample_points,const vector<double> &discrete_potential,const double &value)  {
  assert(is_sorted(begin(sample_points), end(sample_points)));
  assert(value > sample_points.front() || approximately_equal(sample_points.front(), value));
  assert(value < sample_points.back() || approximately_equal(sample_points.back(), value));
  assert(sample_points.size() == discrete_potential.size());
  // TODO: Test if this gives the correct points for the interpolation
  auto lower = floor((value - sample_points[0]) / (sample_points[1] - sample_points[0]));
  auto upper = lower+1;
  assert(lower < upper);
  cout << "Indices Lower:" << lower <<endl;
  return linear_interpolation(value, sample_points[lower], sample_points[upper],
                              discrete_potential[lower], discrete_potential[upper]);
}

double MetadynamicsMetropolisSolver::linear_interpolation(const double &x,const double &x_lower,const double &x_upper,const double &y_lower,const double &y_upper) {
  assert(x_lower < x_upper);
  assert(x > x_lower || approximately_equal(x, x_lower));
  assert(x < x_upper || approximately_equal(x, x_upper));
  auto a =y_lower + (x - x_lower)*(y_upper - y_lower) / (x_upper - x_lower);

  return y_lower + (x - x_lower)*(y_upper - y_lower) / (x_upper - x_lower);
}
