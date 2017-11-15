// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/monitor.h"
#include "jams/core/utils.h"
#include "jams/core/error.h"
#include "jams/core/globals.h"
#include "jams/monitors/magnetisation.h"
#include "jams/monitors/magnetisation_rate.h"
#include "jams/monitors/structurefactor.h"
#include "jams/monitors/torque.h"
#include "jams/monitors/energy.h"
#include "jams/monitors/spin_pumping.h"
#include "jams/monitors/spin_temperature.h"
#include "jams/monitors/boltzmann.h"
#include "jams/monitors/smr.h"
#include "jams/monitors/vtu.h"
#include "jams/monitors/hdf5.h"
#include "jams/monitors/xyz.h"
#include "jams/monitors/binary.h"
#include "jams/monitors/skyrmion.h"

Monitor::Monitor(const libconfig::Setting &settings)
: output_step_freq_(jams::config_optional<int>(settings, "output_steps", jams::default_monitor_output_steps)),
  convergence_is_on_(settings.exists("convergence")),
  convergence_tolerance_(jams::config_optional<double>(settings, "convergence", jams::default_monitor_convergence_tolerance)),
  convergence_stderr_(0.0),
  convergence_burn_time_(jams::config_optional<double>(settings, "t_burn", 0.0))  // amount of time to discard before calculating convegence stats
 {
   output->write("  output_steps: %d (s)\n", output_step_freq_);
   if (convergence_is_on_) {
     ::output->write("  convergence tolerance: %f\n", convergence_tolerance_);
     ::output->write("  t_burn: %e (s)\n", convergence_burn_time_);
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
