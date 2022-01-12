//
// Created by ioannis charalampidis on 20/12/2021.
//

#include "cvar_skyrmion_center_coordinate.h"
#include "jams/metadynamics/skyrmion_center_cv.h"
#include <jams/core/globals.h>
#include "jams/maths/functions.h"
#include <libconfig.h++>
#include <jams/interface/config.h>

#include "jams/core/lattice.h"
#include "jams/helpers/consts.h"
#include "jams/core/physics.h"

#include <algorithm>

//****** Class Initialisation ******
jams::CVarSkyrmionCoreCoordinate::CVarSkyrmionCoreCoordinate(const libconfig::Setting &settings) {
  //settings
//  auto component = config_required<std::string>(settings, "component");
  std::string component = config_required<std::string>(settings, "component");

  if (component == "x" || component == "y") {
	name_ = name_ + component;
  } else {
	throw std::runtime_error(" The Skyrmion Core Coordinate Direction "
							 "Passed is Invalid");
  }

  if (component == "x") {
	value_returned_x = true;
  } else
	value_returned_x = false;

  periodic_x_ = lattice->is_periodic(0);
  periodic_y_ = lattice->is_periodic(1);

  auto bottom_left = lattice->get_unitcell().matrix() * Vec3{0.0, 0.0, 0.0};
  auto bottom_right = lattice->get_unitcell().matrix() * Vec3{double(lattice->size(0)), 0.0, 0.0};
  auto top_left = lattice->get_unitcell().matrix() * Vec3{0.0, double(lattice->size(1)), 0.0};
  auto top_right = lattice->get_unitcell().matrix() * Vec3{double(lattice->size(0)), double(lattice->size(1)), 0.0};

  auto bounds_x = std::minmax({bottom_left[0], bottom_right[0], top_left[0], top_right[0]});
  auto bounds_y = std::minmax({bottom_left[1], bottom_right[1], top_left[1], top_right[1]});
  skyrmion_core_threshold_ = -0.5;
  space_remapping();

}

//******** Public Overridden Functions ***************
double jams::CVarSkyrmionCoreCoordinate::value() {
  return CachingCollectiveVariable::value();
}

double jams::CVarSkyrmionCoreCoordinate::calculate_expensive_value() {
  return skyrmion_center_of_mass_coordinate();
}

double jams::CVarSkyrmionCoreCoordinate::spin_move_trial_value(int i,
															   const Vec3 &spin_initial,
															   const Vec3 &spin_trial) {

  double trial_coordinate = CachingCollectiveVariable::value();

  if (spin_crossed_threshold(spin_initial, spin_trial, skyrmion_core_threshold_)) {
	trial_coordinate = skyrmion_center_of_mass_coordinate();
  }
  return trial_coordinate;
}

std::string jams::CVarSkyrmionCoreCoordinate::name() {
  return name_;
}


//********* Private Class Functions **************

void jams::CVarSkyrmionCoreCoordinate::space_remapping() {
  // The remapping is done in direct (fractional) space rather than real space
  // because it allows us to handle non-square lattices.

  // find maximum extent of the system for normalisation
  cylinder_remapping_x_.resize(globals::num_spins);
  cylinder_remapping_y_.resize(globals::num_spins);

  // We need to remove the third dimension from the lattice matrix because
  // the remapping is in 2D only.
  //
  // NOTE: This means that we are assuming lattice vectors a,b are in the
  // x,y plane and c is BOTH out of the plane and orthogonal to a,b. i.e.
  // it must be a vector along z. We do a check here for safety.
  auto c_unit_vec = normalize(lattice->get_unitcell().c());
  assert(approximately_zero(c_unit_vec[0], 1e-8)
			 && approximately_zero(c_unit_vec[1], 1e-8)
			 && approximately_equal(c_unit_vec[2], 1.0, 1e-8));

  Mat3 W = lattice->get_unitcell().matrix();
  W[0][2] = 0.0;
  W[1][2] = 0.0;
  W[2][2] = 1.0;

  // map 2D space into a cylinder with y as the axis
  double x_max = lattice->size(0);
  auto y_max = lattice->size(1);
  for (auto i = 0; i < globals::num_spins; ++i) {
	auto r = inverse(W) * lattice->atom_position(i);

	if (periodic_x_) {
	  auto theta_x = (r[0] / x_max) * (kTwoPi);

	  auto x = (x_max / (kTwoPi)) * cos(theta_x);
	  auto y = r[1];
	  auto z = (x_max / (kTwoPi)) * sin(theta_x);
	  cylinder_remapping_x_[i] = Vec3{x, y, z};
	} else {
	  cylinder_remapping_x_[i] = r;
	}
	// map 2D space into a cylinder with x as the axis
	if (periodic_y_) {
	  auto theta_y = (r[1] / y_max) * (kTwoPi);

	  auto x = r[0];
	  auto y = (y_max / (kTwoPi)) * cos(theta_y);
	  auto z = (y_max / (kTwoPi)) * sin(theta_y);

	  cylinder_remapping_y_[i] = Vec3{x, y, z};
	} else {
	  cylinder_remapping_y_[i] = r;
	}
  }
}

double jams::CVarSkyrmionCoreCoordinate::skyrmion_center_of_mass_coordinate() {
  using namespace globals;
  double coordinate_x = 0.0;
  double coordinate_y = 0.0;

  Mat3 W = lattice->get_unitcell().matrix();
  W[0][2] = 0.0;
  W[1][2] = 0.0;
  W[2][2] = 1.0;

  Vec3 tube_center_of_mass_x = {0.0, 0.0, 0.0};
  Vec3 tube_center_of_mass_y = {0.0, 0.0, 0.0};

  int num_core_spins = 0;
  for (auto i = 0; i < num_spins; ++i) {
	if (globals::s(i, 2) < skyrmion_core_threshold_) {
	  tube_center_of_mass_x += cylinder_remapping_x_[i];
	  tube_center_of_mass_y += cylinder_remapping_y_[i];
	  num_core_spins++;
	}
  }

  if (periodic_x_) {
	double theta_x = atan2(-tube_center_of_mass_x[2], -tube_center_of_mass_x[1]) + kPi;
	coordinate_x = theta_x * lattice->size(0) / (kTwoPi);
  } else {
	coordinate_x = tube_center_of_mass_x[0] / double(num_core_spins);
  }

  if (periodic_y_) {
	double theta_y = atan2(-tube_center_of_mass_y[2], -tube_center_of_mass_y[1]) + kPi;
	coordinate_y = theta_y * lattice->size(1) / (kTwoPi);
  } else {
	coordinate_y = tube_center_of_mass_y[1] / double(num_core_spins);
  }
  //TODO: Here I return only x or y --> based on the bool passed initially
  if (value_returned_x) {

	return coordinate_x;
  } else
	return coordinate_y;

}

bool jams::CVarSkyrmionCoreCoordinate::spin_crossed_threshold(const Vec3 &s_initial,
															  const Vec3 &s_final,
															  const double &threshold) {
  return (s_initial[2] <= threshold && s_final[2] > threshold) || (s_initial[2] > threshold && s_final[2] <= threshold);
}


