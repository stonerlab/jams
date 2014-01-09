// Copyright 2014 Joseph Barker. All rights reserved.

#include "physics/fieldcool.h"

#include <libconfig.h++>

#include <cmath>
#include <iomanip>

#include "core/globals.h"

void FieldCoolPhysics::init(libconfig::Setting &phys) {
  using namespace globals;

  output.write("  * Field cooled physics module\n");

  initTemp = phys["InitialTemperature"];
  finalTemp = phys["FinalTemperature"];

  for (int i = 0; i < 3; ++i) {
    initField[i] = phys["InitialField"][i];
    finalField[i] = phys["FinalField"][i];
  }

  coolTime = phys["CoolTime"];

  for (int i = 0; i < 3; ++i) {
    globals::h_app[i] += initField[i];
  }

  if (phys.exists("TSteps") == true) {
    TSteps = phys["TSteps"];
    deltaT = (initTemp-finalTemp)/TSteps;
    for (int i = 0; i < 3; ++i) {
        deltaH[i] = (initField[i]-finalField[i])/TSteps;
    }
    t_step = coolTime/TSteps;
    t_eq = config.lookup("sim.t_eq");
    stepToggle = true;
  } else {
    stepToggle = false;
  }
  globalTemperature = initTemp;
  initialised = true;
}

FieldCoolPhysics::~FieldCoolPhysics() {
}

void FieldCoolPhysics::run(const double realtime, const double dt) {
  using namespace globals;

  if (realtime > t_eq) {
    if (stepToggle == true) {
      int stepCount = (realtime-t_eq)/t_step;
      if (stepCount < TSteps+1) {
        globalTemperature = initTemp-stepCount*deltaT;
      }
    } else {
      double fieldRate[3];
      for (int i = 0; i < 3; ++i) {
        fieldRate[i] = ((finalField[i]-initField[i])*dt)/coolTime;
      }
      const double tempRate = ((finalTemp-initTemp)*dt)/coolTime;

      if (realtime < coolTime) {
        for (int i = 0; i < 3; ++i) {
          globals::h_app[i] += fieldRate[i];
        }
        globalTemperature += tempRate;
      }
    }
  }
}

void FieldCoolPhysics::monitor(const double realtime, const double dt)
{}
