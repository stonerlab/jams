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

using namespace std;

void MetadynamicsMetropolisSolver::initialize(const libconfig::Setting &settings) {
  MetropolisMCSolver::initialize(settings);
  intialise_potential_histograms();
  cout << "Metadynamics" << "\n";
  //initialise output files ?
  energy_barrier_file_.open(jams::output::full_path_filename("energy_barrier_metadynamics.tsv"));
}

void MetadynamicsMetropolisSolver::run() {
  MetropolisMCSolver::run();

  if (iteration_ % 100 == 0) {
    magnetisation_ = total_magnetisation_calculation();
	  cout << magnetisation_[2] / globals::num_spins << "\n";
	  insert_gaussian(magnetisation_[2] / globals::num_spins, gaussian_amplitude_, gaussian_width_, sample_points_1d_, potential_1D_);
  }
  if (iteration_ % 500 == 0){
    auto barrier = calculate_energy_difference(potential_1D_);
    cout <<"Iteration: "<< iteration_<< "    energy Barrier: " << barrier << "\n";
    energy_barrier_file_ << iteration_ << "	" << barrier << endl;
  }
}

double MetadynamicsMetropolisSolver::energy_difference(const int spin_index,
                                                       const Vec3 &initial_Spin,
                                                       const Vec3 &final_Spin) {

  return MetropolisMCSolver::energy_difference(spin_index, initial_Spin, final_Spin)
         + potential_difference(spin_index, initial_Spin, final_Spin);
}

double MetadynamicsMetropolisSolver::potential_difference(const int spin_index,const Vec3 &initial_Spin,const Vec3 &final_Spin) {
  magnetisation_ = total_magnetisation_calculation();
  auto initial_potential = interpolated_potential(sample_points_1d_, potential_1D_, magnetisation_[2] / globals::num_spins);

  trial_magnetisation_calculation(magnetisation_, initial_Spin, final_Spin);
  auto trial_potential = interpolated_potential(sample_points_1d_, potential_1D_, trial_magnetisation_[2] / globals::num_spins);

  return trial_potential - initial_potential;
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

void MetadynamicsMetropolisSolver::intialise_potential_histograms()  {
  potential_1D_.resize(sample_points_1d_.size(),0.0);
  potential_2D_.resize(sample_points_2d_.size(),sample_points_m_perpendicular_);
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
  //cout << "Indices Lower:" << lower <<endl; //need to check why why and why
  return linear_interpolation(value, sample_points[lower], sample_points[upper],
                              discrete_potential[lower], discrete_potential[upper]);
}

double MetadynamicsMetropolisSolver::linear_interpolation(const double &x,const double &x_lower,const double &x_upper,const double &y_lower,const double &y_upper) {
  assert(x_lower < x_upper);
  assert(x > x_lower || approximately_equal(x, x_lower));
  assert(x < x_upper || approximately_equal(x, x_upper));

  return y_lower + (x - x_lower)*(y_upper - y_lower) / (x_upper - x_lower);
}
Vec3 MetadynamicsMetropolisSolver::total_magnetisation_calculation() {
  Vec3 m={0, 0, 0};
  for (auto i = 0; i < globals::num_spins; ++i) {
	  for (auto n = 0; n < 3; ++n) {
	    m[n] += globals::s(i, n);
  	}
  }
  return m;
}
Vec3 MetadynamicsMetropolisSolver::trial_magnetisation_calculation(const Vec3 &current_magnetisation, const Vec3 &initial_spin, const Vec3 trial_spin) {
  return current_magnetisation - initial_spin + trial_spin;
}
double MetadynamicsMetropolisSolver::calculate_energy_difference(const vector<double> &potential) {
  const auto margin = potential.size()/4;
  const double max = *max_element(potential.begin()+margin, potential.end()-margin);
  const double min = *min_element(potential.begin()+margin, potential.end()-margin);
  return max - min;
}
