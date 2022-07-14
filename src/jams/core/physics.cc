// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstddef>
#include <string>

#include <libconfig.h++>
#include <jams/helpers/defaults.h>

#include "jams/core/physics.h"

#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/error.h"
#include "jams/core/lattice.h"
#include "jams/physics/empty.h"
#include "jams/physics/field_cool.h"
#include "jams/physics/fmr.h"
#include "jams/physics/mean_first_passage_time.h"
#include "jams/physics/square_field_pulse.h"
#include "jams/physics/two_temperature_model.h"
#include "jams/physics/ping.h"
#include "jams/physics/pinned_boundaries.h"
#include "jams/physics/flips.h"

using namespace std;

Physics::Physics(const libconfig::Setting &physics_settings) :
    Base(physics_settings),
    temperature_(0.0),
    applied_field_{0.0, 0.0, 0.0}
{

  cout << "  " << name() << " physics\n";

  // initialise temperature
  temperature_ = 0.0;
  if (!physics_settings.lookupValue("temperature", temperature_)) {
    jams_warning("No temperature specified in input - assuming 0.0");
  }

  // initialise applied field
  Vec3 field = {0.0, 0.0, 0.0};
  if (physics_settings.exists("applied_field")) {
    if (!physics_settings["applied_field"].isArray() || !(physics_settings["applied_field"].getLength() == 3)) {
      jams_die("Setting 'applied_field' must be an array of length 3.");
    }
    for (int n = 0; n != 3; ++n) {
      field[n] = physics_settings["applied_field"][n];
    }
  }
  applied_field_ = field;

  output_step_freq_ = jams::config_optional<int>(physics_settings, "output_steps", jams::defaults::monitor_output_steps);

  if (physics_settings.exists("initial_state")) {
    libconfig::Setting& state_settings = physics_settings["initial_state"];
    if (!state_settings["origin"].isArray() || !(state_settings["origin"].getLength() == 3)) {
      jams_die("Setting 'initial_state.origin' must be an array of length 3.");

    }

	if (state_settings.exists("relative_x")) {
	  relative_x_ = state_settings["relative_x"];
	  cout << "Relative Skyrmion X position = " << relative_x_ <<"\n";
	}

	if (state_settings.exists("relative_y")) {
	  relative_y_ = state_settings["relative_y"];
	  cout << "Relative Skyrmion Y position = " << relative_y_ <<"\n";
	}


    Vec3 origin;
    for (int i = 0; i < 3; ++i) {
	  if (i == 0) {
		origin[i] = (state_settings["origin"][i]);
		origin[i] = origin[i] * relative_x_;
	  }
	  else if (i == 1){
		origin[i] = (state_settings["origin"][i]);
		origin[i] = origin[i] * relative_y_;
	  }
	  else {
		origin[i] = state_settings["origin"][i];
	  }
    }
    double radius = state_settings["radius"];

    for (int i = 0; i < globals::num_spins; ++i) {
      Vec3 pos = lattice->displacement(lattice->atom_position(i),origin);

      if (pos[0]*pos[0] + pos[1]*pos[1] < radius*radius) {
        globals::s(i,2) = -globals::s(i,2);
      }
    }

  }
  if (physics_settings.exists("perfect_skyrmion")){
	libconfig::Setting& state_settings = physics_settings["perfect_skyrmion"];
	if (!state_settings["origin"].isArray() || !(state_settings["origin"].getLength() == 3)) { // I guess get the origin of the skyrmion centre
	  jams_die("Setting 'initial_state.origin' must be an array of length 3.");
	}
	double perfect_radius = state_settings["perfect_skyrmion_radius"];

	Vec3 origin; // push back the origin indices
	for (int i = 0; i < 3; ++i) {
	  origin[i] = state_settings["origin"][i];
	}

	if (state_settings.exists("relative_x")) {
	  relative_x_ = state_settings["relative_x"];
	  origin[0] = origin[0]* relative_x_;
	  cout << "Relative ('Perfect') Skyrmion X position = " << relative_x_ <<"\n";
	}

	if (state_settings.exists("relative_y")) {
	  relative_y_ = state_settings["relative_y"];
	  origin[1] = origin[1]* relative_y_;
	  cout << "Relative ('Perfect') Skyrmion y position = " << relative_y_ <<"\n";
	}
	cout << "Skyrmion Initialed at: [" << origin[0]<< "," << origin[1] << "," << origin[2] <<"]" << "\n";

	for (int i = 0; i < globals::num_spins; ++i) {
	  Vec3 displacement = lattice->displacement(lattice->atom_position(i),origin);
		double x_distance = displacement[0];
		double y_distance = displacement[1];
		double z_distance = displacement[2];
		double theta_spherical = theta_calculation(x_distance, y_distance, z_distance, perfect_radius);
		Vec3 cartesian_mag_directions = spherical_to_cartesian(x_distance,y_distance,theta_spherical);
		//push back the skyrmion in the lattice
		for(auto ii = 0; ii < cartesian_mag_directions.size(); ++ii) {
		  assert(ii < 3 || ii == 2);
		  globals::s(i, ii) = cartesian_mag_directions[ii];

	}
	}
  }

}

Physics* Physics::create(const libconfig::Setting &settings) {

  std::string module_name = jams::defaults::physics_module;
  settings.lookupValue("module", module_name);
  module_name = lowercase(module_name);

  if (module_name == "empty") {
    return new EmptyPhysics(settings);
  }

  if (module_name == "fmr") {
    return new FMRPhysics(settings);
  }

  if (module_name == "mean-first-passage-time") {
    return new MFPTPhysics(settings);
  }

  if (module_name == "two-temperature-model") {
    return new TTMPhysics(settings);
  }

  if (module_name == "square-field-pulse") {
    return new SquarePhysics(settings);
  }

  if (module_name == "field-cool") {
    return new FieldCoolPhysics(settings);
  }

  if (module_name == "ping") {
    return new PingPhysics(settings);
  }

  if (module_name == "pinned_boundaries") {
    return new PinnedBoundariesPhysics(settings);
  }

  if (module_name == "flip") {
    return new FlipsPhysics(settings);
  }

  throw std::runtime_error("unknown physics " + module_name);
}

double Physics::theta_calculation(const double &x, const double &y, const double &z, const double& r_skyrmion) {
  double r = sqrt(x*x + y*y + z*z);
  double theta_r = 2*(atan2(r_skyrmion,r)); //use atan2 to take into account the correct quadrant base on the signs of x,y
  return theta_r;
}

Vec3 Physics::spherical_to_cartesian(double x, double y, const double theta) {
  Vec3 cart_cordinates;
  double phi = atan2(y,x);
  cart_cordinates[0]= cos(phi)*sin(theta);
  cart_cordinates[1]= sin(phi)*sin(theta);
  cart_cordinates[2]= cos(theta);
  return cart_cordinates;
}