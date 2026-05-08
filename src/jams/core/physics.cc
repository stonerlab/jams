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
#include "jams/interface/config.h"
#include "jams/physics/empty.h"
#include "jams/physics/field_cool.h"
#include "jams/physics/fmr.h"
#include "jams/physics/induced_spin_pulse.h"
#include "jams/physics/mean_first_passage_time.h"
#include "jams/physics/square_field_pulse.h"
#include "jams/physics/two_temperature_model.h"
#include "jams/physics/ping.h"
#include "jams/physics/pinned_boundaries.h"
#include "jams/physics/flips.h"
#include <jams/helpers/exception.h>

Physics::Physics(const libconfig::Setting &physics_settings) :
    Base(physics_settings),
    temperature_(0.0),
    applied_field_{0.0, 0.0, 0.0}
{

  std::cout << "  " << name() << " physics\n";

  // initialise temperature
  temperature_ = 0.0;
  if (!physics_settings.lookupValue("temperature", temperature_)) {
    jams_warning("No temperature specified in input - assuming 0.0");
  }

  // initialise applied field
  jams::Vec<double, 3> field = {0.0, 0.0, 0.0};
  if (physics_settings.exists("applied_field")) {
    field = jams::read_vec_setting<double, 3>(physics_settings["applied_field"], "applied_field");
  }
  applied_field_ = field;

  output_step_freq_ = jams::config_optional<int>(physics_settings, "output_steps", jams::defaults::monitor_output_steps);

  if (physics_settings.exists("initial_state")) {
    libconfig::Setting& state_settings = physics_settings["initial_state"];
    const auto origin = jams::read_vec_setting<double, 3>(state_settings["origin"], "origin");
    const double radius = jams::read_numeric_setting<double>(state_settings["radius"], "radius");

    for (int i = 0; i < globals::num_spins; ++i) {
      jams::Vec<double, 3> pos = globals::lattice->displacement(
          globals::lattice->lattice_site_position_cart(i), origin);

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

  if (module_name == "induced-spin-pulse") {
    return new InducedSpinPulsePhysics(settings);
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
