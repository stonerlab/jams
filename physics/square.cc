// Copyright 2014 Joseph Barker. All rights reserved.

#include "physics/square.h"

#include <libconfig.h++>

#include <cmath>

#include "core/globals.h"

void SquarePhysics::init(libconfig::Setting &phys) {
  using namespace globals;

  output.write("  * Square physics module\n");

  PulseDuration = phys["PulseDuration"];
  PulseTotal    = phys["PulseTotal"];
  PulseCount = 1;

  for (int i = 0; i < 3; ++i) {
    FieldStrength[i] = phys["FieldStrength"][i];
  }

  initialised = true;
}

SquarePhysics::~SquarePhysics() {
}

void SquarePhysics::run(const double realtime, const double dt) {
  using namespace globals;

  double eqtime = config.lookup("sim.t_eq");

  if ((realtime > eqtime) && ((realtime-eqtime) > (PulseDuration*PulseCount))) {
    PulseCount++;
  }

  if ((realtime > eqtime) &&
      ((realtime-eqtime) < (PulseDuration*(PulseTotal)))) {
    if (PulseCount%2 == 0) {
      for (int i = 0; i < 3; ++i) {
        globals::h_app[i] = -FieldStrength[i];
      }
    } else {
      for (int i = 0; i < 3; ++i) {
        globals::h_app[i] = FieldStrength[i];
      }
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      globals::h_app[i] = 0.0;
    }
  }
}

void SquarePhysics::monitor(const double realtime, const double dt) {
}
