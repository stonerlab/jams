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
  sample_points_y_ = linear_space_creation(0,lattice->rmax()[1],histogram_step_size_); // TODO : pass it from the lattice size x and y from the confiq file
  potential_2d_.resize(sample_points_x_.size(),std::vector<double>(sample_points_x_.size(),0.0));
  r_com.resize(lattice->num_materials(), {0.0, 0.0, 0.0});
  create_center_of_mass_mapping();
  skyrmion_outfile.open(jams::output::full_path_filename("sky.tsv"));
}

//-------OVERWRITTEN FUNCTIONS ---------//

void jams::SkyrmionCenterCV::output() {
  if (solver->iteration()% 1000 == 0){
    skyrmion_output();
  }
  if (solver->iteration()%10000 == 0){
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
  create_center_of_mass_mapping(); //TODO : if statement to recalculate this every x-times to avoid numerical drift
  calc_center_of_mass(r_com,tube_x,tube_y);

  for (int i = 0; i < sample_points_y_.size(); ++i) {
	for (int ii = 0; ii < sample_points_x_.size(); ++ii) {
	  potential_2d_[i][ii] +=  gaussian_2D(reinterpret_cast<const double &>(r_com[0][1]), sample_points_y_[i], r_com[0][0], sample_points_y_[ii], gaussian_amplitude_*relative_amplitude); // TODO :  r_com

	}
  }

}
double jams::SkyrmionCenterCV::current_potential() {
//  TODO : calculate the potential the "current potential" (calculate the COM) used only for the tempering
 return interpolated_2d_potential(r_com[0][1],r_com[0][0]); //TODO : think carefully  which material_number on  the r_com and trial_r_com should be used and why
}
double jams::SkyrmionCenterCV::potential_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final){
  double initial_potential = current_potential();
  double trial_potential;
  trial_center_of_mass(spin_final,i); // Final spin and spin index "i" passed from the monte_carlo_class
  trial_potential = interpolated_2d_potential(trial_r_com[0][1],trial_r_com[0][1]);
  return trial_potential - initial_potential;
}
void jams::SkyrmionCenterCV::spin_update(int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
//  TODO: if the move is accepted update the two cylinders (first convert to cylindrical cordinates).
// just use the tubes_update function inside an if-statement
}

//-------SKYRMION CENTER OF MASS FUNCTIONS---------//

void jams::SkyrmionCenterCV::create_center_of_mass_mapping() { // the original function has been broken into two functions. Now this function calls the tubes_update(spin_index) function
                                                               // this allows me to use it to update the tubes if the monte carlo move is accepted. Also used to update the trial_tubes.
  using namespace globals;

  tube_x.resize(num_spins, {0.0, 0.0, 0.0});
  tube_y.resize(num_spins, {0.0, 0.0, 0.0});

  for (int n = 0; n < num_spins; ++n) {
	tubes_update(n);
  }
}
void jams::SkyrmionCenterCV::tubes_update(const int &spin) { // I broke it down so I can actually update the tubes when the MC move is accepted
  using namespace globals;
  double i, j, i_max, j_max, r_i, r_j, theta_i, theta_j, x, y, z;

  i = lattice->atom_position(spin)[0];
  j = lattice->atom_position(spin)[1];

  i_max = lattice->rmax()[0];
  j_max = lattice->rmax()[1];

  r_i = i_max / (kTwoPi);
  r_j = j_max / (kTwoPi);

  theta_i = (i / i_max) * (kTwoPi);
  theta_j = (j / j_max) * (kTwoPi);

  x = r_i * cos(theta_i);
  y = j;
  z = r_i * sin(theta_i);

  tube_x[spin] = {x, y, z};

  x = i;
  y = r_j * cos(theta_j);
  z = r_j * sin(theta_j);

  tube_y[spin] = {x, y, z};

}
void jams::SkyrmionCenterCV::calc_center_of_mass(std::vector<Vec3> &r_com_passed, std::vector<Vec3 > &tube_x_passed, std::vector<Vec3 > &tube_y_passed){ //const double &threshold) {
  using namespace globals;
  using namespace std;
  // TODO: make the x and y PBC individually optional

// assert(tube_x.size() > 0);
//  assert(tube_y.size() > 0);

  const int num_types = lattice->num_materials();

  std::vector<Vec3 > tube_x_com(num_types, {0.0, 0.0, 0.0});
  std::vector<Vec3 > tube_y_com(num_types, {0.0, 0.0, 0.0});
  int r_count[num_types];
  tube_x_passed = tube_x_com;
  tube_y_passed = tube_x_com;

  for (auto type = 0; type < num_types; ++type) {
	r_count[type] = 0;
  }

  for (auto i = 0; i < num_spins; ++i) {
	auto type = lattice->atom_material_id(i);
//	if (s(i, 2)*type_norms[type] > threshold) {
//	  tube_x_com[type] += tube_x[i];
//	  tube_y_com[type] += tube_y[i];
//	  r_count[type]++;
//	}
  }

  for (auto type = 0; type < num_types; ++type) {
	r_com[type] /= static_cast<double>(r_count[type]);
  }

  for (auto type = 0; type < num_types; ++type) {
	double theta_i = atan2(-tube_x_passed[type][2], -tube_x_passed[type][0]) + kPi;
	double theta_j = atan2(-tube_y_passed[type][2], -tube_y_passed[type][1]) + kPi;

	r_com_passed[type][0] = (theta_i*lattice->rmax()[0]/(kTwoPi));
	r_com_passed[type][1] = (theta_j*lattice->rmax()[1]/(kTwoPi));
	r_com_passed[type][2] = 0.0;
  }

}// Original function is modified. So it can be used
                                                                                                                                                      //for both the trial and current tubes and r_com
void jams::SkyrmionCenterCV::trial_center_of_mass(Vec3 trial_spin, int spin_index) {
auto trial_tube_x = tube_x;
auto trial_tube_y = tube_y;

trial_tube_x[spin_index][0]=trial_spin[0];
trial_tube_y[spin_index][0]=trial_spin[1];
calc_center_of_mass(trial_r_com,trial_tube_x,trial_tube_y);

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
  using namespace globals;

  double x, y;

  const double x_size = lattice->rmax()[0];
  const double y_size = lattice->rmax()[1];

  skyrmion_outfile << std::setw(12) << std::scientific << solver->time();
  skyrmion_outfile << std::setw(16) << std::fixed << solver->physics()->temperature();

  for (double threshold : thresholds) {
//	std::vector<Vec3 > r_com(lattice->num_materials(), {0.0, 0.0, 0.0});
	calc_center_of_mass(r_com,tube_x,tube_y);

	int r_count[lattice->num_materials()];
	double radius_gyration[lattice->num_materials()];

	for (auto i = 0; i < lattice->num_materials(); ++i) {
	  r_count[i] = 0;
	  radius_gyration[i] = 0.0;
	}

	for (auto i = 0; i < num_spins; ++i) {
	  auto type = lattice->atom_material_id(i);
	  if (s(i, 2)*type_norms[type] > threshold) {
		x = lattice->atom_position(i)[0] - r_com[type][0];
		x = x - nint(x / x_size) * x_size;  // min image convention
		y = lattice->atom_position(i)[1] - r_com[type][1];
		y = y - nint(y / y_size) * y_size;  // min image convention
		radius_gyration[type] += x*x + y*y;
		r_count[type]++;
	  }
	}

	for (auto n = 0; n < lattice->num_materials(); ++n) {
	  radius_gyration[n] = sqrt(radius_gyration[n]/static_cast<double>(r_count[n]));
	}

	for (auto n = 0; n < lattice->num_materials(); ++n) {
	  if (r_count[n] == 0) {
		for (auto i = 0; i < 5; ++i) {
		  skyrmion_outfile << std::setw(16) << 0.0;
		}
	  } else {
		for (auto i = 0; i < 3; ++i) {
		  skyrmion_outfile << std::setw(16) << r_com[n][i]*lattice->parameter();
		}
		skyrmion_outfile << std::setw(16) << radius_gyration[n]*lattice->parameter() << std::setw(16) << (2.0/sqrt(2.0))*radius_gyration[n]*lattice->parameter();
	  }
	}
  }

  skyrmion_outfile << "\n";

} // havent spent any time on this TODO: check it