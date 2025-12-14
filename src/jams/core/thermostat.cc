// Copyright 2014 Joseph Barker. All rights reserved.
#include "jams/core/thermostat.h"

#include "jams/interface/config.h"
#include "jams/core/globals.h"
#include "jams/helpers/utils.h"

#include "jams/thermostats/cuda_thermostat_classical.h"
#include "jams/thermostats/thm_bose_einstein_cuda_srk4.h"
#include "jams/thermostats/cuda_thermostat_general_fft.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

#include <string>
#include <stdexcept>
#include <iostream>

Thermostat* Thermostat::create(const std::string &thermostat_name, const jams::Real timestep) {
  std::cout << thermostat_name << " thermostat\n";

  auto temperature = jams::config_required<double>(
      globals::config->lookup("physics"), "temperature");

  // create the selected thermostat
  #if HAS_CUDA
  if (capitalize(thermostat_name) == "CLASSICAL-GPU" || capitalize(thermostat_name) == "LANGEVIN-WHITE-GPU" || capitalize(thermostat_name) == "CUDA_LANGEVIN_WHITE") {
      return new CudaThermostatClassical(temperature, 0.0, timestep, globals::num_spins);
  }
  if (capitalize(thermostat_name) == "QUANTUM-SPDE-GPU" || capitalize(thermostat_name) == "LANGEVIN-BOSE-GPU" ||capitalize(thermostat_name) == "CUDA_LANGEVIN_COTH") {
    return new CudaThermostatQuantumSpde(temperature, 0.0, timestep, globals::num_spins);
  }
  if (capitalize(thermostat_name) == "LANGEVIN-BOSE-SRK4-GPU") {
    return new jams::BoseEinsteinCudaSRK4Thermostat(temperature, 0.0, timestep, globals::num_spins);
  }
  if (capitalize(thermostat_name) == "GENERAL-FFT-GPU" || capitalize(thermostat_name) == "LANGEVIN-LORENTZIAN-GPU" || capitalize(thermostat_name) == "LANGEVIN-ARBITRARY-GPU" ||capitalize(thermostat_name) == "CUDA_LANGEVIN_ARBITRARY") {
    return new CudaThermostatGeneralFFT(temperature, 0.0, timestep, globals::num_spins);
  }
  #endif

    // throw error if the thermostat name is no known
  throw std::runtime_error("unknown thermostat " + thermostat_name);
}
