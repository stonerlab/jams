// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/monitor.h"
#include "core/utils.h"
#include "core/globals.h"
#include "core/solver.h"
#include "monitors/magnetisation.h"
#include "monitors/magnetisation_rate.h"
#include "monitors/structurefactor.h"
#include "monitors/torque.h"
#include "monitors/energy.h"
#include "monitors/spin_pumping.h"
#include "monitors/spin_temperature.h"
#include "monitors/boltzmann.h"
#include "monitors/smr.h"
#include "monitors/vtu.h"
#include "monitors/hdf5.h"
#include "monitors/xyz.h"
#include "monitors/binary.h"
#include "monitors/skyrmion.h"

Monitor::Monitor(const libconfig::Setting &settings)
: is_equilibration_monitor_(false),
  output_step_freq_(100) {
  settings.lookupValue("eq_monitor", is_equilibration_monitor_);
  settings.lookupValue("output_steps", output_step_freq_);

  if (settings.exists("output_steps")) {
    output_step_freq_ = settings["output_steps"];
    output.write("  output_steps: %d (s)\n", output_step_freq_);
  } else {
    ::output.write("  DEFAULT output_steps (100)\n");
    output_step_freq_ = 100; // DEFAULT
  }
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

  if (capitalize(settings["module"]) == "MAGNETISATION_RATE") {
    return new MagnetisationRateMonitor(settings);
  }

  if (capitalize(settings["module"]) == "STRUCTUREFACTOR") {
    return new StructureFactorMonitor(settings);
  }

  if (capitalize(settings["module"]) == "TORQUE") {
    return new TorqueMonitor(settings);
  }

  if (capitalize(settings["module"]) == "ENERGY") {
    return new EnergyMonitor(settings);
  }

  if (capitalize(settings["module"]) == "SPIN_TEMPERATURE") {
    return new SpinTemperatureMonitor(settings);
  }

  if (capitalize(settings["module"]) == "SPINPUMPING") {
    return new SpinPumpingMonitor(settings);
  }

  if (capitalize(settings["module"]) == "BOLTZMANN") {
    return new BoltzmannMonitor(settings);
  }

  if (capitalize(settings["module"]) == "SMR") {
    return new SMRMonitor(settings);
  }

  if (capitalize(settings["module"]) == "VTU") {
    return new VtuMonitor(settings);
  }

  if (capitalize(settings["module"]) == "HDF5") {
    return new Hdf5Monitor(settings);
  }

  if (capitalize(settings["module"]) == "XYZ") {
    return new XyzMonitor(settings);
  }

  if (capitalize(settings["module"]) == "BINARY") {
    return new BinaryMonitor(settings);
  }

  if (capitalize(settings["module"]) == "SKYRMION") {
    return new SkyrmionMonitor(settings);
  }

  jams_error("Unknown monitor specified '%s'", settings["module"].c_str());
  return NULL;
}
