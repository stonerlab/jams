// Copyright 2014 Joseph Barker. All rights reserved.

#include "square_field_pulse.h"

#include <libconfig.h++>

#include <cmath>

#include "jams/core/globals.h"

SquarePhysics::SquarePhysics(const libconfig::Setting &settings)
  : Physics(settings),
  PulseDuration(0),
  PulseCount(0),
  PulseTotal(0),
  FieldStrength(3, 0) {
  PulseDuration = settings["PulseDuration"];
  PulseTotal    = settings["PulseTotal"];
  PulseCount = 1;

  for (int i = 0; i < 3; ++i) {
    FieldStrength[i] = settings["FieldStrength"][i];
  }
}

SquarePhysics::~SquarePhysics() {
}

void SquarePhysics::update(const int &iterations, const double &time, const double &dt) {
  double eqtime = globals::config->lookup("sim.t_eq");

  if ((time > eqtime) && ((time-eqtime) > (PulseDuration*PulseCount))) {
    PulseCount++;
  }

  if ((time > eqtime) &&
      ((time-eqtime) < (PulseDuration*(PulseTotal)))) {
    if (PulseCount%2 == 0) {
      for (int i = 0; i < 3; ++i) {
        applied_field_[i] = -FieldStrength[i];
      }
    } else {
      for (int i = 0; i < 3; ++i) {
        applied_field_[i] = FieldStrength[i];
      }
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      applied_field_[i] = 0.0;
    }
  }
}
