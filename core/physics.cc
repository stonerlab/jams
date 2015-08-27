// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/physics.h"

#include "jblib/containers/vec.h"

#include <libconfig.h++>

#include "core/globals.h"
#include "core/monitor.h"
#include "core/utils.h"

#include "physics/empty.h"
#include "physics/fieldcool.h"
#include "physics/fmr.h"
#include "physics/mfpt.h"
#include "physics/square.h"
#include "physics/ttm.h"


Physics::Physics(const libconfig::Setting &physics_settings) : temperature_(0.0),
    applied_field_(0.0, 0.0, 0.0) {

  // initialise temperature
  temperature_ = 0.0;
  if (!physics_settings.lookupValue("temperature", temperature_)) {
    jams_warning("No temperature specified in input - assuming 0.0");
  }

  // initialise applied field
  jblib::Vec3<double> field(0.0, 0.0, 0.0);
  if (physics_settings.exists("applied_field")) {
    if (!physics_settings["applied_field"].isArray() || !(physics_settings["applied_field"].getLength() == 3)) {
      jams_error("Setting 'applied_field' must be an array of length 3.");
    }
    for (int n = 0; n != 3; ++n) {
      field[n] = physics_settings["applied_field"][n];
    }
  }
  applied_field_ = field;

  if (physics_settings.exists("output_steps")) {
    output_step_freq_ = physics_settings["output_steps"];
  } else {
    jams_warning("No physics output_steps chosen - using default of 100");
    output_step_freq_ = 100;
  }

  jblib::Vec3<double> origin;
  double radius;
  if (physics_settings.exists("initial_state")) {
    libconfig::Setting& state_settings = physics_settings["initial_state"];
    if (!state_settings["origin"].isArray() || !(state_settings["origin"].getLength() == 3)) {
      jams_error("Setting 'initial_state.origin' must be an array of length 3.");
    }
    for (int i = 0; i < 3; ++i) {
      origin[i] = state_settings["origin"][i];
    }
    radius = state_settings["radius"];

    for (int i = 0; i < globals::num_spins; ++i) {
      jblib::Vec3<double> pos = (lattice.position(i)-origin);

      if (pos.x*pos.x + pos.y*pos.y < radius*radius) {
        globals::s(i,2) = -globals::s(i,2);
      }
    }

  }

}

Physics* Physics::create(const libconfig::Setting &settings) {

  if (capitalize(settings["module"]) == "FMR") {
    return new FMRPhysics(settings);
  }

  if (capitalize(settings["module"]) == "MFPT") {
    return new MFPTPhysics(settings);
  }

  if (capitalize(settings["module"]) == "TTM") {
    return new TTMPhysics(settings);
  }

  if (capitalize(settings["module"]) == "SQUARE") {
    return new SquarePhysics(settings);
  }

  if (capitalize(settings["module"]) == "FIELDCOOL") {
    return new FieldCoolPhysics(settings);
  }

  if (capitalize(settings["module"]) == "EMPTY") {
    return new EmptyPhysics(settings);
  }

  jams_error("Unknown physics package selected.");
  return NULL;
}
