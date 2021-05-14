//
// Created by Ioannis Charalampidis on 26/04/2021.
//

#include "jams/metadynamics/skyrmion_center_cv.h"
#include <jams/core/globals.h>
#include <jams/maths/interpolation.h>
#include <libconfig.h++>
#include <fstream>
#include <jams/interface/config.h>
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/helpers/output.h"
#include "jams/helpers/maths.h"
#include <string>
#include <algorithm>

std::vector<double> linear_space_creation(const double &min, const double &max, const double &steps) { //duplicate symbol 'linear_space(double const&, double const&, double const&)' in:
  //CMakeFiles/jams.dir/metadynamics/mz_orthogonal_mz_cv.cc.o
  // CMakeFiles/jams.dir/metadynamics/skyrmion_center_cv.cc.o
  assert(min < max);
  std::vector<double> space;
  double value = min;
  while (value < max + steps) {
	space.push_back(value);
	value += steps;
  }

  return space;
}
jams::SkyrmionCenterCV::SkyrmionCenterCV(const libconfig::Setting &settings) {
  gaussian_amplitude_ = jams::config_required<double>(settings, "gaussian_amplitude");

  // maximum amplitude of inserted gaussians in Joules
  // maximum amplitude of inserted Gaussians in Joules
  // (this can be reduced by tempering in the metadynamics solver)
  gaussian_amplitude_ = jams::config_required<double>(settings, "gaussian_amplitude") / kBohrMagneton;

  // width of the gaussian in units of ??
  gaussian_width_ = jams::config_required<double>(settings, "gaussian_width");
  // discretisation width of the metadynamics potential landscape in units of ??
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
  //Create the 2d_potential landscape with dimension of the lattice points along x and y
  sample_points_x_ = linear_space_creation(0,lattice->rmax()[0],histogram_step_size_);
  sample_points_y_ = linear_space_creation(0,lattice->rmax()[0],histogram_step_size_); // TODO : dont know why rmax()[1] goes only up to 55.5 that's why I use rmax()[0] for y
  potential_2d_.resize(sample_points_x_.size(),std::vector<double>(sample_points_x_.size(),0.0));
  skyrmion_outfile.open(jams::output::full_path_filename("sky_test.tsv"));
  skyrmion_com.open(jams::output::full_path_filename("com_track.tsv"));
  skyrmion_outfile <<"Iteration"<< "	"<< "x" << "	"<< "y" << "	"<< "z" << "\n" ;
  skyrmion_com <<"Iteration"<< "	"<< "x" << "	"<< "y" << "	"<< "z" << "\n" ;
  skyrmion_threshold_ = 0.05;

  space_remapping();
  cached_initial_center_of_mass_ = calc_center_of_mass();
  cached_trial_center_of_mass_ = cached_initial_center_of_mass_;
}

//-------OVERWRITTEN FUNCTIONS ---------//

void jams::SkyrmionCenterCV::output() {
  //if (solver->iteration()% 10 == 0){
    //skyrmion_output();
  //}
  skyrmion_com << solver->iteration()<< "	" <<cached_initial_center_of_mass_[0] << "	" << cached_initial_center_of_mass_[1] << "	" << cached_initial_center_of_mass_[2] << "\n";

  if (solver->iteration()%1000 == 0){
    potential_landscape.open(jams::output::full_path_filename("skyrmion_potential.tsv"));
	for (auto i = 0; i < sample_points_y_.size(); ++i) {
	  for (auto ii = 0; ii < sample_points_x_.size(); ++ii) {
		potential_landscape << sample_points_x_[ii] << "	" << sample_points_y_[i] << "	"
				  << potential_2d_[i][ii] * kBohrMagneton << "\n";
	  }
	}
	potential_landscape.close();
  }

  }

void jams::SkyrmionCenterCV::insert_gaussian(const double &relative_amplitude) {

  for (int i = 0; i < sample_points_y_.size(); ++i) {
	for (int ii = 0; ii < sample_points_x_.size(); ++ii) {
	  potential_2d_[i][ii] +=  gaussian_2D(reinterpret_cast<const double &>(cached_initial_center_of_mass_[1]), sample_points_y_[i], cached_initial_center_of_mass_[0], sample_points_x_[ii], gaussian_amplitude_*relative_amplitude); // TODO :  r_com

	}
  }

}
double jams::SkyrmionCenterCV::current_potential() {
 return interpolated_2d_potential(cached_initial_center_of_mass_[0], cached_initial_center_of_mass_[1]);
}

double jams::SkyrmionCenterCV::potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final){
  double initial_potential = current_potential(); // save it in a gloabal variable
  double trial_potential;

  Vec3 trial_center_of_mass = cached_initial_center_of_mass_;

  if (spin_crossed_threshold(spin_initial, spin_final, skyrmion_threshold_)) {
    trial_center_of_mass = calc_center_of_mass();
    cached_trial_center_of_mass_ = trial_center_of_mass;
  }

  trial_potential = interpolated_2d_potential(trial_center_of_mass[0],trial_center_of_mass[1]);
  return trial_potential - initial_potential;
}
void jams::SkyrmionCenterCV::spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
  // we don't need to check if the threshold was crossed here because
  // cached_trial_center_of_mass_ contains the correct center of mass
  // from the potential_difference() function
  cached_initial_center_of_mass_ = cached_trial_center_of_mass_;
}


Vec3 jams::SkyrmionCenterCV::calc_center_of_mass() {
  using namespace globals;
  using namespace std;

  Vec3 tube_center_of_mass_x = {0.0, 0.0, 0.0};
  Vec3 tube_center_of_mass_y = {0.0, 0.0, 0.0};

  for (auto i = 0; i < num_spins; ++i) {
    if (globals::s(i,2) < skyrmion_threshold_) {
      tube_center_of_mass_x += tube_x_[i];
      tube_center_of_mass_y += tube_y_[i];
    }
  }

	double theta_x = atan2(-tube_center_of_mass_x[2], -tube_center_of_mass_x[0]) + kPi;
	double theta_y = atan2(-tube_center_of_mass_y[2], -tube_center_of_mass_y[1]) + kPi;

	Vec3 center_of_mass = {
      theta_x*lattice->rmax()[0]/(kTwoPi),
      theta_y*lattice->rmax()[1]/(kTwoPi),
      0.0
  };

	return center_of_mass;
}

//---------------Private Functions---------------//

double jams::SkyrmionCenterCV::gaussian_2D(const double &x,const double &x0,const double &y,const double &y0,const double amplitude) const {

  // return gaussian_amplitude_* exp(-(((x - x0) * (x - x0)) + ((y - y0) * (y - y0))) / (2.0 * gaussian_width_ * gaussian_width_)); //to try fix the tempered amplitude
  return amplitude
	  * exp(-(((x - x0) * (x - x0)) + ((y - y0) * (y - y0))) / (2.0 * gaussian_width_ * gaussian_width_));

}
double jams::SkyrmionCenterCV::interpolated_2d_potential(const double &y, const double x) {
//  assert(m <= 1); TODO : think carefylly for appropriate assert checks for x and y .
//  assert(x < 1);
  double lower_y = floor(abs((y - sample_points_y_[0]) / (sample_points_y_[1]
	  - sample_points_y_[0]))); //lower_y index for the discrete_potential
  double upper_y = lower_y + 1;
  double lower_x = floor(abs((x - sample_points_x_[0]) / (sample_points_x_[1]
	  - sample_points_x_[0])));//lower_x index for the discrete_potential
  double upper_x = lower_x + 1;

  assert(lower_y < upper_y);
  assert(lower_x < upper_x);
  //f(x1,y1)=Q(11) , f(x1,y2)=Q(12), f(x2,y1), f(x2,y2)
  double Q11 = potential_2d_[lower_y][lower_x];
  double Q12 = potential_2d_[lower_y][upper_x];
  double Q21 = potential_2d_[upper_y][lower_x];
  double Q22 = potential_2d_[upper_y][upper_x];



  //Interpolate along the x-axis
  double R1 = jams::maths::linear_interpolation(x,
												sample_points_x_[lower_x],
												Q11,
												sample_points_x_[upper_x],
												Q21);
  double R2 = jams::maths::linear_interpolation(x,
												sample_points_x_[lower_x],
												Q12,
												sample_points_x_[upper_x],
												Q22);
  //Interpolate along the y-axis
  return jams::maths::linear_interpolation(y, sample_points_y_[lower_y], R1, sample_points_y_[upper_y], R2);
}
void jams::SkyrmionCenterCV::skyrmion_output() {
//  using namespace globals;
//
//  double x, y;
//
//  const double x_size = lattice->rmax()[0];
//  const double y_size = lattice->rmax()[1];
//
//  skyrmion_outfile << std::setw(12) << std::scientific << solver->time();
//  skyrmion_outfile << std::setw(16) << std::fixed << solver->physics()->temperature();
//
//	calc_center_of_mass();
//
//	int r_count[lattice->num_materials()];
//	double radius_gyration[lattice->num_materials()];
//
//	for (auto i = 0; i < lattice->num_materials(); ++i) {
//	  r_count[i] = 0;
//	  radius_gyration[i] = 0.0;
//	}
//
//	for (auto i = 0; i < num_spins; ++i) {
//	  auto type = lattice->atom_material_id(i);
//	  if (s(i, 2)*type_norms[type] > threshold) {
//		x = lattice->atom_position(i)[0] - r_com[type][0];
//		x = x - nint(x / x_size) * x_size;  // min image convention
//		y = lattice->atom_position(i)[1] - r_com[type][1];
//		y = y - nint(y / y_size) * y_size;  // min image convention
//		radius_gyration[type] += x*x + y*y;
//		r_count[type]++;
//	  }
//	}
//
//	for (auto n = 0; n < lattice->num_materials(); ++n) {
//	  radius_gyration[n] = sqrt(radius_gyration[n]/static_cast<double>(r_count[n]));
//	}
//
//	for (auto n = 0; n < lattice->num_materials(); ++n) {
//	  if (r_count[n] == 0) {
//		for (auto i = 0; i < 5; ++i) {
//		  skyrmion_outfile << std::setw(16) << 0.0;
//		}
//	  } else {
//		for (auto i = 0; i < 3; ++i) {
//		  skyrmion_outfile << std::setw(16) << r_com[n][i]*lattice->parameter();
//		}
//		skyrmion_outfile << std::setw(16) << radius_gyration[n]*lattice->parameter() << std::setw(16) << (2.0/sqrt(2.0))*radius_gyration[n]*lattice->parameter();
//	  }
//	}
//  }
//
//  skyrmion_outfile << "\n";

} // haven't spent any time on this TODO: check it

void jams::SkyrmionCenterCV::space_remapping() {

  // find maximum extent of the system for normalisation

  tube_x_.resize(globals::num_spins);
  tube_y_.resize(globals::num_spins);

  // map 2D space into a cylinder with y as the axis
  auto x_max = lattice->rmax()[0];
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r = lattice->atom_position(i);

    auto theta_x = (r[0] / x_max) * (kTwoPi);

    auto x = (x_max / (kTwoPi)) * cos(theta_x);
    auto y = r[1];
    auto z = (x_max / (kTwoPi)) * sin(theta_x);

    tube_x_[i] = {x, y, z};
  }

  // map 2D space into a cylinder with x as the axis
  auto y_max = lattice->rmax()[1];
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto r = lattice->atom_position(i);

    auto theta_y = (r[1] / y_max) * (kTwoPi);

    auto x = r[0];
    auto y = y_max * cos(theta_y);
    auto z = y_max * sin(theta_y);

    tube_y_[i] = {x, y, z};
  }
}

bool jams::SkyrmionCenterCV::spin_crossed_threshold(const Vec3 &s_initial,
                                                    const Vec3 &s_final,
                                                    const double &threshold) {
  return (s_initial[2] <= threshold && s_final[2] > threshold) || (s_initial[2] > threshold && s_final[2] <= threshold);
}
