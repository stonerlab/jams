// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/monitor.h"
#include "core/utils.h"
#include "core/globals.h"
#include "core/solver.h"
#include "monitors/magnetisation.h"
#include "monitors/structurefactor.h"
#include "monitors/torque.h"
#include "monitors/anisotropy_energy.h"
#include "monitors/energy.h"
#include "monitors/boltzmann.h"
#include "monitors/vtu.h"
#include "monitors/xyz.h"

Monitor::Monitor(const libconfig::Setting &settings)
: is_equilibration_monitor_(false),
  output_step_freq_(100) {
  settings.lookupValue("eq_monitor", is_equilibration_monitor_);
  settings.lookupValue("output_steps", output_step_freq_);
}

bool Monitor::is_updating(const int &iteration) const {
  if (iteration % output_step_freq_ == 0) {
    return true;
  }
  return false;
}

Monitor* Monitor::create(const libconfig::Setting &settings) {
  if (capitalize(settings["module"]) == "MAGNETISATION") {
    return new MagnetisationMonitor(settings);
  }

  if (capitalize(settings["module"]) == "STRUCTUREFACTOR") {
    return new StructureFactorMonitor(settings);
  }

  if (capitalize(settings["module"]) == "TORQUE") {
    return new TorqueMonitor(settings);
  }

  if (capitalize(settings["module"]) == "ANISOTROPY_ENERGY") {
    return new AnisotropyEnergyMonitor(settings);
  }

  if (capitalize(settings["module"]) == "ENERGY") {
    return new EnergyMonitor(settings);
  }

  if (capitalize(settings["module"]) == "BOLTZMANN") {
    return new BoltzmannMonitor(settings);
  }

  if (capitalize(settings["module"]) == "VTU") {
    return new VtuMonitor(settings);
  }

  if (capitalize(settings["module"]) == "XYZ") {
    return new XyzMonitor(settings);
  }

  jams_error("Unknown monitor specified '%s'", settings["module"].c_str());
  return NULL;
}
