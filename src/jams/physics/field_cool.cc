// Copyright 2014 Joseph Barker. All rights reserved.

#include "field_cool.h"

#include <libconfig.h++>

#include "jams/core/globals.h"

FieldCoolPhysics::FieldCoolPhysics(const libconfig::Setting &settings)
  : Physics(settings),
  initField(3, 0.0),
  finalField(3, 0.0),
  deltaH(3, 0.0),
  initTemp(0.0),
  finalTemp(0.0),
  coolTime(0.0),
  TSteps(0),
  deltaT(0),
  t_step(0),
  t_eq(0),
  integration_time_step_(0.0),
  stepToggle(false),
  initialized(false) {
  using namespace globals;

  initTemp = settings["InitialTemperature"];
  finalTemp = settings["FinalTemperature"];

  integration_time_step_ = ::config->lookup("sim.t_step");

  for (int i = 0; i < 3; ++i) {
    initField[i] = settings["InitialField"][i];
    finalField[i] = settings["FinalField"][i];
  }

  coolTime = settings["CoolTime"];

  for (int i = 0; i < 3; ++i) {
    applied_field_[i] += initField[i];
  }

  if (settings.exists("TSteps") == true) {
    TSteps = settings["TSteps"];
    deltaT = (initTemp-finalTemp)/TSteps;
    for (int i = 0; i < 3; ++i) {
        deltaH[i] = (initField[i]-finalField[i])/TSteps;
    }
    t_step = coolTime/TSteps;
    t_eq = config->lookup("sim.t_eq");
    stepToggle = true;
  } else {
    stepToggle = false;
  }
  temperature_ = initTemp;
  initialized = true;
}

void FieldCoolPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;

  if (time > t_eq) {
    if (stepToggle == true) {
      int stepCount = (time-t_eq)/t_step;
      if (stepCount < TSteps+1) {
        temperature_ = initTemp-stepCount*deltaT;
      }
    } else {
      double fieldRate[3];
      for (int i = 0; i < 3; ++i) {
        fieldRate[i] = ((finalField[i]-initField[i])*integration_time_step_)/coolTime;
      }
      const double tempRate = ((finalTemp-initTemp)*integration_time_step_)/coolTime;

      if (time < coolTime) {
        for (int i = 0; i < 3; ++i) {
          applied_field_[i] += fieldRate[i];
        }
        temperature_ += tempRate;
      }
    }
  }
}
