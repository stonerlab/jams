// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/monitor.h"
#include "jams/core/utils.h"
#include "jams/core/error.h"
#include "jams/core/globals.h"
#include "jams/monitors/magnetisation.h"
#include "jams/monitors/unitcell-magnetisation.h"
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
: output_step_freq_(100),
  convergence_is_on_(false),
  convergence_tolerance_(0.001),
  convergence_stderr_(0.0),
  convergence_burn_time_(0.0)  // amount of time to discard before calculating convegence stats
 {
  settings.lookupValue("output_steps", output_step_freq_);

  if (settings.exists("output_steps")) {
    output_step_freq_ = settings["output_steps"];
    output->write("  output_steps: %d (s)\n", output_step_freq_);
  } else {
    ::output->write("  DEFAULT output_steps (100)\n");
    output_step_freq_ = 100; // DEFAULT
  }

  if (settings.exists("convergence")) {
    convergence_is_on_ = true;

    convergence_tolerance_ = settings["convergence"];
    ::output->write("  convergence tolerance: %f\n", convergence_tolerance_);

    if (settings.exists("t_burn")) {
      convergence_burn_time_ = settings["t_burn"];
    } else {
      ::output->write("  DEFAULT t_burn (0.001*t_sim)\n");
      convergence_burn_time_ = 0.001 * double(config->lookup("sim.t_sim"));     // DEFAULT
    }

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

  if (capitalize(settings["module"]) == "UNITCELL-MAGNETISATION") {
    return new UnitcellMagnetisationMonitor(settings);
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
