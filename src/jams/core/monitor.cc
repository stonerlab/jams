// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/globals.h"
#include "jams/core/monitor.h"
#include "jams/monitors/binary.h"
#include "jams/monitors/boltzmann.h"
#include "jams/monitors/energy.h"
#include "jams/monitors/hdf5.h"
#include "jams/monitors/magnetisation.h"
#include "jams/monitors/magnetisation_rate.h"
#include "jams/monitors/skyrmion.h"
#include "jams/monitors/smr.h"
#include "jams/monitors/spin_pumping.h"
#include "jams/monitors/spin_temperature.h"
#include "jams/monitors/spectrum_fourier.h"
#include "jams/monitors/spectrum_general.h"
#include "jams/monitors/torque.h"
#include "jams/monitors/vtu.h"
#include "jams/monitors/xyz.h"

#ifdef HAS_CUDA
#include "jams/monitors/cuda_spectrum_general.h"
#endif

using namespace std;

Monitor::Monitor(const libconfig::Setting &settings)
: Base(settings),
  output_step_freq_(jams::config_optional<int>(settings, "output_steps", jams::default_monitor_output_steps)),
  convergence_is_on_(settings.exists("convergence")),
  convergence_tolerance_(jams::config_optional<double>(settings, "convergence", jams::default_monitor_convergence_tolerance)),
  convergence_stderr_(0.0),
  convergence_burn_time_(jams::config_optional<double>(settings, "t_burn", 0.0))  // amount of time to discard before calculating convegence stats
{
  cout << "  " << name() << " monitor\n";
  cout << "    output_steps" << output_step_freq_ << "\n";

   if (convergence_is_on_) {
     cout << "    convergence tolerance" << convergence_tolerance_ << "\n";
     cout << "    t_burn" << convergence_burn_time_ << "\n";
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

  if (capitalize(settings["module"]) == "STRUCTUREFACTOR"
  || capitalize(settings["module"]) == "SPECTRUM_FOURIER") {
    return new SpectrumFourierMonitor(settings);
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

  if (capitalize(settings["module"]) == "SCATTERING-FUNCTION"
  || capitalize(settings["module"]) == "SPECTRUM_GENERAL") {
#ifdef HAS_CUDA
    if (solver->is_cuda_solver()) {
      return new CudaSpectrumGeneralMonitor(settings);
    }
#endif
    return new SpectrumGeneralMonitor(settings);
  }

  throw std::runtime_error("unknown monitor " + std::string(settings["module"].c_str()));
}
