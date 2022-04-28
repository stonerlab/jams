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
	  cout << "Relative Skyrmion X position = " << relative_y_ <<"\n";
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

  if (module_name == "flip") {
    return new FlipsPhysics(settings);
  }

  throw std::runtime_error("unknown physics " + module_name);
}
