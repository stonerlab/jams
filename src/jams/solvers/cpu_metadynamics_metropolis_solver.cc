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

using namespace std;

void MetadynamicsMetropolisSolver::initialize(const libconfig::Setting &settings) {
  MetropolisMCSolver::initialize(settings);
  intialise_potential_histograms();
  physical_region_indices(sample_points_1d_,lower_limit_index,upper_limit_index);
  energy_barrier_file_.open(jams::output::full_path_filename("energy_barrier_metadynamics.tsv"));
  potential_1d_file_.open(jams::output::full_path_filename("potential_1d_metadynamics.tsv"));
}

void MetadynamicsMetropolisSolver::run() {
  // update the total magnetisation to avoid the possibility of drift with accumulated errors
  magnetisation_ = total_magnetisation_calculation();
  MetropolisMCSolver::run();
  if (iteration_ % 100 == 0) {

	  insert_gaussian(magnetisation_[2] / globals::num_spins, gaussian_amplitude_, gaussian_width_, sample_points_1d_, potential_1D_);
  }
  if (iteration_ % 500 == 0){
    auto barrier = calculate_energy_difference(potential_1D_);
    cout <<"Iteration: "<< iteration_<< "    energy Barrier: " << barrier << "\n";
    energy_barrier_file_ << iteration_ << "	" << barrier << endl;
  }
  if (iteration_ % 1000 == 0){
    potential_1d_print(potential_1d_file_,lower_limit_index,upper_limit_index);
  }
}

double MetadynamicsMetropolisSolver::energy_difference(const int spin_index,
                                                       const Vec3 &initial_Spin,
                                                       const Vec3 &final_Spin) {

  return MetropolisMCSolver::energy_difference(spin_index, initial_Spin, final_Spin)
         + potential_difference(spin_index, initial_Spin, final_Spin);
}

double MetadynamicsMetropolisSolver::potential_difference(const int spin_index,const Vec3 &initial_Spin,const Vec3 &final_Spin) {

  const Vec3 initial_magnetisation = magnetisation_;
  auto initial_potential = interpolated_potential(sample_points_1d_, potential_1D_, initial_magnetisation[2] / globals::num_spins);

  const Vec3 trial_magnetisation = magnetisation_ - initial_Spin + final_Spin;
  auto trial_potential = interpolated_potential(sample_points_1d_, potential_1D_, trial_magnetisation[2] / globals::num_spins);

  return trial_potential - initial_potential;
}

std::vector<double> MetadynamicsMetropolisSolver::linear_space(const double &min,const double &max,const double &step) {
  assert(min < max);
  vector<double> space;
  double value = min;
  while (value < max+step) {
    space.push_back(value);
    value += step;
  }

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
  return jams::maths::linear_interpolation(value,
                              sample_points[lower], discrete_potential[lower],
                              sample_points[upper], discrete_potential[upper]);
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

double MetadynamicsMetropolisSolver::calculate_energy_difference(const vector<double> &potential) {
  const auto margin = potential.size()/4;
  const double max = *max_element(potential.begin()+margin, potential.end()-margin);
  const double min = *min_element(potential.begin()+margin, potential.end()-margin);
  return max - min;
}

void MetadynamicsMetropolisSolver::accept_move(const int spin_index,
                                               const Vec3 &initial_spin,
                                               const Vec3 &final_spin) {
  MetropolisMCSolver::accept_move(spin_index, initial_spin, final_spin);
  cout << "mag: " << magnetisation_[2]/globals::num_spins << " S_in: " << initial_spin[2] << " S_final: " << final_spin[2] << "\n";
  magnetisation_ - initial_spin + final_spin;
}

void MetadynamicsMetropolisSolver::physical_region_indices(const std::vector<double>& points,int &lower_limit,int &upper_limit) {

  for (int i = 0; i < points.size(); ++i) {
	if (floats_are_equal(lower_limit, points[i])) {
	  cout << endl << "Lower Index: " << i << " Element at index " << points[i] << endl;
	  lower_limit = i;

	  break;
	}
  }
  for (int ii = lower_limit; ii < points.size(); ++ii) {
	if (floats_are_equal(upper_limit, points[ii])) {
	  cout << " Upper Index: " << ii << " Element at index " << points[ii] << endl;
	  upper_limit = ii;
	  break;
	}
  }
}
bool MetadynamicsMetropolisSolver::floats_are_equal(const double &x, const double &y, const double epsilon) {
  return abs(x - y) < epsilon;
}
void MetadynamicsMetropolisSolver::potential_1d_print(ofstream &potential_1d_file, const double &lower_vector_index, const double &upper_vector_index) {
  for (auto i = lower_vector_index; i < upper_vector_index + 1; ++i) {
	potential_1d_file << sample_points_1d_[i] << "	" << potential_1D_[i] << endl;
  }

}
