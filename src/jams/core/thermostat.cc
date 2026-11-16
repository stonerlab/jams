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
    cudaEventCreateWithFlags(&consumed_, cudaEventDisableTiming);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
#endif
}

Thermostat::~Thermostat() {
#if HAS_CUDA
  if (done_) {
    cudaEventDestroy(done_);
  }
  if (consumed_) {
    cudaEventDestroy(consumed_);
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

Thermostat* Thermostat::create(const std::string &thermostat_name, const double timestep,
                               const std::optional<double> stage_interval) {
  std::cout << thermostat_name << " thermostat\n";

  const auto temperature = jams::config_required<double>(
      globals::config->lookup("physics"), "temperature");
  const auto name = capitalize(thermostat_name);
  // Preserve the timestep precision of existing factory callers.
  const auto legacy_timestep = static_cast<jams::Real>(timestep);

  #if HAS_CUDA
  const double coloured_interval = stage_interval.value_or(legacy_timestep);
  if (name == "CLASSICAL-GPU" || name == "LANGEVIN-WHITE-GPU" || name == "CUDA_LANGEVIN_WHITE") {
    return new CudaThermostatClassical(temperature, 0.0, legacy_timestep, globals::num_spins);
  }
  if (name == "QUANTUM-SPDE-GPU" || name == "LANGEVIN-BOSE-GPU" || name == "CUDA_LANGEVIN_COTH") {
    return new CudaThermostatQuantumSpde(temperature, 0.0, coloured_interval, globals::num_spins);
  }
  if (name == "GENERAL-FFT-GPU" || name == "LANGEVIN-LORENTZIAN-GPU"
      || name == "LANGEVIN-ARBITRARY-GPU" || name == "CUDA_LANGEVIN_ARBITRARY") {
    return new CudaThermostatGeneralFFT(temperature, 0.0, coloured_interval, globals::num_spins);
  }
  #endif

  if (stage_interval) {
    throw std::runtime_error("thermostat " + thermostat_name + " is unsupported with a stage sampling interval");
  }

  if (name == "CLASSICAL-CPU" || name == "LANGEVIN-WHITE-CPU") {
    return new ThermostatClassical(temperature, 0.0, legacy_timestep, globals::num_spins);
  }
  if (name == "QUANTUM-SPDE-CPU" || name == "LANGEVIN-BOSE-CPU" || name == "CPU_LANGEVIN_COTH") {
    return new ThermostatQuantumSpde(temperature, 0.0, legacy_timestep, globals::num_spins);
  }

  #if HAS_CUDA
  if (name == "LANGEVIN-BOSE-SRK4-GPU") {
    return new jams::BoseEinsteinCudaSRK4Thermostat(temperature, 0.0, legacy_timestep, globals::num_spins);
  }
  #endif

  throw std::runtime_error("unknown thermostat " + thermostat_name);
}
