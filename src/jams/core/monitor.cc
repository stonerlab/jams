// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/globals.h"
#include "jams/core/monitor.h"
#include "jams/monitors/binary.h"
#include "jams/monitors/boltzmann.h"
#include "jams/monitors/energy.h"
#include "jams/monitors/field.h"
#include "jams/monitors/topological_charge_finite_diff.h"
#include "jams/monitors/topological_charge_geometrical_def.h"
#include "jams/monitors/hdf5.h"
#include "jams/monitors/magnetisation.h"
#include "jams/monitors/magnetisation_layers.h"
#include "jams/monitors/magnetisation_rate.h"
#include "jams/monitors/magnon_spectrum.h"
#include "jams/monitors/skyrmion.h"
#include "jams/monitors/smr.h"
#include "jams/monitors/neutron_scattering.h"
#include "jams/monitors/neutron_scattering_no_lattice.h"
#include "jams/monitors/cuda_neutron_scattering_no_lattice.h"
#include "jams/monitors/spectrum_fourier.h"
#include "jams/monitors/spectrum_general.h"
#include "jams/monitors/spin_correlation.h"
#include "jams/monitors/spin_pumping.h"
#include "jams/monitors/spin_temperature.h"
#include "jams/monitors/torque.h"
#include "jams/monitors/unitcell_average.h"
#include "jams/monitors/vtu.h"
#include "jams/monitors/xyz.h"

#ifdef HAS_CUDA
  #include "jams/monitors/neutron_scattering_no_lattice.h"
  #include "jams/monitors/cuda_spin_current.h"
  #include "jams/monitors/cuda_thermal_current.h"
#endif

#define DEFINED_MONITOR(name, type, settings) \
{ \
  if (lowercase(settings["module"]) == name) { \
    return new type(settings); \
  } \
}

Monitor::Monitor(const libconfig::Setting &settings)
: Base(settings),
  output_step_freq_(
          jams::config_optional<int>(settings, "output_steps", jams::defaults::monitor_output_steps)),
  convergence_status_(ConvergenceStatus::kDisabled),
  convergence_tolerance_(
          jams::config_optional<double>(settings, "convergence", jams::defaults::monitor_convergence_tolerance)),
  convergence_stderr_(
          0.0),
  convergence_burn_time_(jams::config_optional<double>(settings, "t_burn", 0.0))
{
  std::cout << "  " << name() << " monitor\n";
  std::cout << "    output_steps " << output_step_freq_ << "\n";

   if (settings.exists("convergence")) {
     convergence_status_ = ConvergenceStatus::kNotConverged;
     std::cout << "    convergence tolerance" << convergence_tolerance_ << "\n";
     std::cout << "    t_burn" << convergence_burn_time_ << "\n";
   }
}

bool Monitor::is_updating(const int &iteration) {
  if (iteration % output_step_freq_ == 0) {
    return true;
  }
  return false;
}

Monitor* Monitor::create(const libconfig::Setting &settings) {
  DEFINED_MONITOR("binary", BinaryMonitor, settings);
  DEFINED_MONITOR("boltzmann", BoltzmannMonitor, settings);
  DEFINED_MONITOR("energy", EnergyMonitor, settings);
  DEFINED_MONITOR("field", FieldMonitor, settings);
  DEFINED_MONITOR("topological-charge-finite-diff", TopologicalFiniteDiffChargeMonitor, settings);
  DEFINED_MONITOR("topological-charge-geometrical-def", TopologicalGeometricalDefMonitor, settings);
  DEFINED_MONITOR("hdf5", Hdf5Monitor, settings);
  DEFINED_MONITOR("magnetisation", MagnetisationMonitor, settings);
  DEFINED_MONITOR("magnetisation-layers", MagnetisationLayersMonitor, settings);
  DEFINED_MONITOR("magnetisation-rate", MagnetisationRateMonitor, settings);
  DEFINED_MONITOR("skyrmion", SkyrmionMonitor, settings);
  DEFINED_MONITOR("smr", SMRMonitor, settings);
  DEFINED_MONITOR("magnon-spectrum", MagnonSpectrumMonitor, settings);
  DEFINED_MONITOR("neutron-scattering", NeutronScatteringMonitor, settings);
  DEFINED_MONITOR("neutron-scattering-no-lattice", NeutronScatteringNoLatticeMonitor, settings);
  DEFINED_MONITOR("spectrum-fourier", SpectrumFourierMonitor, settings);
  DEFINED_MONITOR("spectrum-general-cpu", SpectrumGeneralMonitor, settings);
  DEFINED_MONITOR("spin-correlation", SpinCorrelationMonitor, settings);
  DEFINED_MONITOR("spin-pumping", SpinPumpingMonitor, settings);
  DEFINED_MONITOR("spin-temperature", SpinTemperatureMonitor, settings);
  DEFINED_MONITOR("torque", TorqueMonitor, settings);
  DEFINED_MONITOR("unitcell-average", UnitcellAverageMonitor, settings);
  DEFINED_MONITOR("vtu", VtuMonitor, settings);
  DEFINED_MONITOR("xyz", XyzMonitor, settings);

#ifdef HAS_CUDA
  DEFINED_MONITOR("spin-current", CudaSpinCurrentMonitor, settings);
  DEFINED_MONITOR("thermal-current", CudaThermalCurrentMonitor, settings);
  DEFINED_MONITOR("cuda-neutron-scattering-no-lattice", CudaNeutronScatteringNoLatticeMonitor, settings);
#endif

  throw std::runtime_error("unknown monitor " + std::string(settings["module"].c_str()));
}
