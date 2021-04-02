//
// Created by ioannis charalampidis on 25/11/2020.
//

#include "mz_orthogonal_mz_cv.h"
#include <jams/core/globals.h>
#include <jams/maths/interpolation.h>
#include <libconfig.h++>
#include "fstream"
#include <jams/interface/config.h>
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

std::vector<double> linear_space(const double &min, const double &max, const double &step) {
  assert(min < max);
  std::vector<double> space;
  double value = min;
  while (value < max + step) {
	space.push_back(value);
	value += step;
  }

  return space;
}

jams::MzOrthogonalMzCV::MzOrthogonalMzCV(const libconfig::Setting &settings) {

  gaussian_amplitude_ = jams::config_required<double>(settings, "gaussian_amplitude");
  // ---------------------------------------------------------------------------
  // config settings
  // ---------------------------------------------------------------------------

  // maximum amplitude of inserted gaussians in Joules
  // (this can be reduced by tempering in the metadynamics solver)
  gaussian_amplitude_ = jams::config_required<double>(settings, "gaussian_amplitude")/ kBohrMagneton;

  // width of the gaussian in units of mz
  gaussian_width_ = jams::config_required<double>(settings, "gaussian_width");
  // discretisation width of the metadynamics potential landscape in units of mz
  histogram_step_size_ = jams::config_required<double>(settings, "histogram_step_size");

  // ---------------------------------------------------------------------------
  // validate settings
  // ---------------------------------------------------------------------------

  // If histogram_step_size does not divide evenly into the range -1 -> 1 then
  // we will be missing either the start of the end point of the physical range.
  if (!approximately_equal(std::remainder(2.0, histogram_step_size_), 0.0)) {
	throw std::runtime_error("Invalid value of histogram_step_size: "
							 "histogram_step_size must divide into 2.0 with no remainder");
  }

  sample_points_mz_ = linear_space(-1.0, 1.0, histogram_step_size_);
  sample_points_mz_perpendicular_ = linear_space(0, 1.0, histogram_step_size_);
  potential_2d_.resize(sample_points_mz_.size(), std::vector<double>(sample_points_mz_perpendicular_.size(), 0.0));
  metadynamics_simulation_parameters.open(jams::output::full_path_filename("_parameters.tsv"));
  metadynamics_simulation_parameters.open(jams::output::full_path_filename("potential_difference.tsv"));

  magnetisation_ = calculate_total_magnetisation();
  assert(magnetisation_[2] / globals::num_spins <= 1);
}

void jams::MzOrthogonalMzCV::output() {

  if (solver->iteration() % 1000 == 0) {
	metadynamics_simulation_parameters << solver->iteration() << "	" << energy_barrier_calculation() << "\n";
  if (solver->iteration() % 10000 == 0) {
	metadynamics_simulation_parameters << solver->iteration() << "	" << energy_barrier_calculation()*kBohrMagneton << "\n";
  }

  if (solver->iteration() % 100000 == 0) {
	potential.open(jams::output::full_path_filename("_potential2d.tsv"));
  if (solver->iteration() % 1000000 == 0) {
	potential.open(jams::output::full_path_filename("potential.tsv"));
	for (auto i = 0; i < sample_points_mz_.size(); ++i) {
	  for (auto ii = 0; ii < sample_points_mz_perpendicular_.size(); ++ii) {
		potential << sample_points_mz_perpendicular_[ii] << "	" << sample_points_mz_[i] << "	"
				  << potential_2d_[i][ii] << "\n";\
				  << potential_2d_[i][ii]*kBohrMagneton << "\n";\
	  }
	}
	potential.close();
  }
}

double jams::MzOrthogonalMzCV::current_potential() {
  return interpolated_2d_potential(magnetisation_[2] / globals::num_spins, mz_perpendicular(magnetisation_));
}

double jams::MzOrthogonalMzCV::potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
  Vec3 initial_magnetisation = magnetisation_;
  assert(magnetisation_[2] / globals::num_spins <= 1);
  const double initial_m_perpendicular = mz_perpendicular(initial_magnetisation);
  assert(initial_m_perpendicular <= 1);
  auto initial_potential =interpolated_2d_potential(initial_magnetisation[2] / globals::num_spins, initial_m_perpendicular);

  Vec3 trial_magnetisation =
	  magnetisation_ - spin_initial + spin_final; // TODO: check which is normalised and which is not
  assert(trial_magnetisation[2] / globals::num_spins <= 1);
  const double trial_m_perpendicular = mz_perpendicular(trial_magnetisation);
  assert (trial_m_perpendicular <= 1);
  auto trial_potential = interpolated_2d_potential(trial_magnetisation[2] / globals::num_spins, trial_m_perpendicular);

  return trial_potential - initial_potential;
}

void jams::MzOrthogonalMzCV::spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
  magnetisation_ = magnetisation_ - spin_initial + spin_final;

}

void jams::MzOrthogonalMzCV::insert_gaussian(const double &relative_amplitude) {
  mz_perpendicular_ = mz_perpendicular(magnetisation_);
  for (int i = 0; i < sample_points_mz_.size(); ++i) {
	for (int ii = 0; ii < sample_points_mz_perpendicular_.size(); ++ii) {
	  potential_2d_[i][ii] += relative_amplitude * gaussian_2D(magnetisation_[2] / globals::num_spins,sample_points_mz_[i],mz_perpendicular_,sample_points_mz_perpendicular_[ii], gaussian_amplitude_);

	}
  }
}

double jams::MzOrthogonalMzCV::collective_coordinate() {
  return magnetisation_[2] / globals::num_spins;
}

double jams::MzOrthogonalMzCV::interpolated_2d_potential(const double &m, const double m_p) {
  assert(m <= 1);
  assert(m_p < 1);
  double lower_m = floor(abs((m - sample_points_mz_[0]) / (sample_points_mz_[1]
	  - sample_points_mz_[0]))); //lower_y index for the discrete_potential
  double upper_m = lower_m + 1;
  double lower_m_p = floor(abs((m_p - sample_points_mz_perpendicular_[0]) / (sample_points_mz_perpendicular_[1]
	  - sample_points_mz_perpendicular_[0])));//lower_x index for the discrete_potential
  double upper_m_p = lower_m_p + 1;

  assert(lower_m < upper_m);
  assert(lower_m_p < upper_m_p);
  //f(x1,y1)=Q(11) , f(x1,y2)=Q(12), f(x2,y1), f(x2,y2)
  double Q11 = potential_2d_[lower_m][lower_m_p];
  double Q12 = potential_2d_[lower_m][upper_m_p];
  double Q21 = potential_2d_[upper_m][lower_m_p];
  double Q22 = potential_2d_[upper_m][upper_m_p];



  //Interpolate along the x-axis
  double R1 = jams::maths::linear_interpolation(m_p,
												sample_points_mz_perpendicular_[lower_m_p],
												Q11,
												sample_points_mz_perpendicular_[upper_m_p],
												Q21);
  double R2 = jams::maths::linear_interpolation(m_p,
												sample_points_mz_perpendicular_[lower_m_p],
												Q12,
												sample_points_mz_perpendicular_[upper_m_p],
												Q22);
  //Interpolate along the y-axis
  return jams::maths::linear_interpolation(m, sample_points_mz_[lower_m], R1, sample_points_mz_[upper_m], R2);
}

Vec3 jams::MzOrthogonalMzCV::calculate_total_magnetisation() {
  Vec3 m = {0, 0, 0};
  for (auto i = 0; i < globals::num_spins; ++i) {
	for (auto n = 0; n < 3; ++n) {
	  m[n] += globals::s(i, n);
	}
  }
  return m;
}

double jams::MzOrthogonalMzCV::mz_perpendicular(Vec3 &magnetisation) {
  return sqrt(((magnetisation[0] / globals::num_spins) * (magnetisation[0] / globals::num_spins))
				  + ((magnetisation[1] / globals::num_spins) * (magnetisation[1] / globals::num_spins)));
}

double jams::MzOrthogonalMzCV::gaussian_2D(const double &x, const double &x0, const double &y, const double &y0, const double amplitude) const {
 // return gaussian_amplitude_* exp(-(((x - x0) * (x - x0)) + ((y - y0) * (y - y0))) / (2.0 * gaussian_width_ * gaussian_width_)); //to try fix the tempered amplitude
  return amplitude* exp(-(((x - x0) * (x - x0)) + ((y - y0) * (y - y0))) / (2.0 * gaussian_width_ * gaussian_width_));
}
double jams::MzOrthogonalMzCV::energy_barrier_calculation() {
  auto min = 1.0;
  auto max = 0.0;
  for (auto i = 0; i < potential_2d_.size(); ++i) {
	for (auto ii = 0; ii < potential_2d_[i].size(); ++ii) {
	  if (potential_2d_[i][ii] > max) {
		max = potential_2d_[i][ii];
	  }
	  if (potential_2d_[i][ii] < min) {
		min = potential_2d_[i][ii];
	  }

	  // assert(floats_are_equal(min, max) || min < max);
	}
  }
  return max - min;
}


