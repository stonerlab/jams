// Copyright 2014 Joseph Barker. All rights reserved.

#include "two_temperature_model.h"

#include <libconfig.h++>

#include <cmath>
#include <iomanip>
#include <string>

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"

#include "jblib/math/functions.h"

TTMPhysics::TTMPhysics(const libconfig::Setting &settings)
  : Physics(settings),
  pulseWidth(0),
  pulseFluence(0),
  pulseStartTime(0),
  pumpTemp(0.0),
  electronTemp(0.0),
  phononTemp(0.0),
  sinkTemp(0.0),
  reversingField(3, 0.0),
  Ce(7.0E02),
  Cl(3.0E06),
  G(17.0E17),
  Gsink(17.0E14),
  TTMFile() {
  using namespace globals;

  phononTemp = settings["InitialTemperature"];
  electronTemp = phononTemp;

  sinkTemp = phononTemp;

  const libconfig::Setting& laserPulseConfig =
    ::config->lookup("physics.laserPulses");

  const int nLaserPulses = laserPulseConfig.getLength();

  pulseWidth.resize(nLaserPulses);
  pulseFluence.resize(nLaserPulses);
  pulseStartTime.resize(nLaserPulses);

  for (int i = 0; i != nLaserPulses; ++i) {
    pulseWidth(i) = laserPulseConfig[i]["width"];
    pulseFluence(i) = laserPulseConfig[i]["fluence"];
    pulseFluence(i) = pumpPower(pulseFluence(i));
    pulseStartTime(i) = laserPulseConfig[i]["t_start"];
  }

  // if these settings don't exist, the defaults will be left in
  settings.lookupValue("Ce", Ce);
  settings.lookupValue("Cl", Cl);
  settings.lookupValue("Gep", G);
  settings.lookupValue("Gps", Gsink);

  for (int i = 0; i < 3; ++i) {
    reversingField[i] = settings["ReversingField"][i];
  }

  std::string fileName = "_ttm.dat";
  fileName = seedname+fileName;
  TTMFile.open(fileName.c_str());

  TTMFile << std::setprecision(8);
  TTMFile << "# t [s]\tT_el [K]\tT_ph [K]\tLaser [arb/]\n";
}

TTMPhysics::~TTMPhysics() {
  TTMFile.close();
}

void TTMPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;
  using namespace jblib;

double real_dt = dt/kGyromagneticRatio;

  for (int i = 0; i < 3; ++i) {
    applied_field_[i] = reversingField[i];
  }

  pumpTemp = 0.0;
  for (int i = 0, iend = pulseFluence.size(); i != iend; ++i) {
    const double relativeTime = (time-pulseStartTime(i));
    if ((relativeTime > 0.0) && (relativeTime <= 10*pulseWidth(i))) {
      pumpTemp = pumpTemp
        + pulseFluence(i)
        *exp(-square((relativeTime-3*pulseWidth(i))/(pulseWidth(i))));
    }
  }

  electronTemp = electronTemp
    + ((-G*(electronTemp-phononTemp)+pumpTemp)*real_dt)/(Ce*electronTemp);
  phononTemp   = phononTemp
    + ((G*(electronTemp-phononTemp)-Gsink*(phononTemp-sinkTemp))*real_dt)/(Cl);

  temperature_ = electronTemp;

  if (iterations%output_step_freq_ == 0) {
    TTMFile << time << "\t" << electronTemp << "\t"
    << phononTemp << "\t" << pumpTemp << "\n";
  }
}
