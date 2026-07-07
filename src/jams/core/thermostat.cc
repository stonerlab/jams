// Copyright 2014 Joseph Barker. All rights reserved.
#include "jams/core/thermostat.h"

#include "jams/interface/config.h"
#include "jams/core/globals.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"

#include "jams/thermostats/thermostat_classical.h"
#include "jams/thermostats/thermostat_quantum_spde.h"

#include "jams/thermostats/cuda_thermostat_classical.h"
#include "jams/thermostats/thm_bose_einstein_cuda_srk4.h"
#include "jams/thermostats/cuda_thermostat_general_fft.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

#include <string>
#include <stdexcept>
#include <iostream>

#if HAS_CUDA
namespace {

bool cuda_device_available_for_event_creation() {
  int device_count = 0;
  const auto status = cudaGetDeviceCount(&device_count);
  if (status == cudaSuccess) {
    return device_count > 0;
  }

  cudaGetLastError();
  return false;
}

}  // namespace
#endif

Thermostat::Thermostat(const jams::Real &temperature,
                       const jams::Real &sigma,
                       const jams::Real timestep,
                       const int num_spins)
    : temperature_(temperature),
      temperature_profile_(jams::ThermostatTemperatureProfile::from_config(
          *globals::config, num_spins, temperature)),
      sigma_(num_spins, 3),
      noise_(num_spins, 3) {
  (void)sigma;
  (void)timestep;
  sigma_.zero();
  noise_.zero();

#if HAS_CUDA
  if (cuda_device_available_for_event_creation()) {
    cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
#endif
}

void Thermostat::set_temperature(const jams::Real T) {
  temperature_profile_.set_uniform_temperature(T);
  temperature_ = T;
}

void Thermostat::update_in_parallel() {
  throw jams::unimplemented_error("Thermostat::update_in_parallel");
}

Thermostat* Thermostat::create(const std::string &thermostat_name, const jams::Real timestep) {
  std::cout << thermostat_name << " thermostat\n";

  auto temperature = jams::config_required<double>(
      globals::config->lookup("physics"), "temperature");

  // create the selected thermostat
  if (capitalize(thermostat_name) == "CLASSICAL-CPU"
      || capitalize(thermostat_name) == "LANGEVIN-WHITE-CPU") {
    return new ThermostatClassical(temperature, 0.0, timestep, globals::num_spins);
  }
  if (capitalize(thermostat_name) == "QUANTUM-SPDE-CPU"
      || capitalize(thermostat_name) == "LANGEVIN-BOSE-CPU"
      || capitalize(thermostat_name) == "CPU_LANGEVIN_COTH") {
    return new ThermostatQuantumSpde(temperature, 0.0, timestep, globals::num_spins);
  }

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
