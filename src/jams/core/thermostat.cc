// Copyright 2014 Joseph Barker. All rights reserved.
#include "jams/core/thermostat.h"

#include "jams/interface/config.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"

#include "jams/thermostats/thermostat_classical.h"
#include "jams/thermostats/thermostat_quantum_spde.h"

#include "jams/thermostats/cuda_thermostat_classical.h"
#include "jams/thermostats/thm_bose_einstein_cuda_srk4.h"
#include "jams/thermostats/cuda_thermostat_general_fft.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

#include <string>
#include <cmath>
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

std::vector<double> jams::thermal_field_coupling(
    const MultiArray<Real, 1>& gyro, const ThermalTorque torque) {
  const auto alpha = globals::alpha.host_span();
  const auto inv_mu = globals::inv_mus.host_span();
  const auto rate = gyro.host_span();
  std::vector<double> coupling(rate.size());
  for (std::size_t i = 0; i < rate.size(); ++i) {
    if (alpha[i] == 0.0 || inv_mu[i] == 0.0) continue;

    // For unit-spin LLG the thermal field drives perpendicular precession
    // and damping torques. Their variances add, giving 1 + alpha^2. The
    // dissipative mobility is alpha * gyro_eff / mu for both local equations.
    // Retarded precession uses the bare rate to parameterise the kernel.
    const double damping = alpha[i];
    const double torque_variance = torque == ThermalTorque::llg ? 1.0 + damping * damping : 1.0;
    coupling[i] = damping * inv_mu[i] / (rate[i] * torque_variance);
    if (!std::isfinite(coupling[i]) || coupling[i] < 0.0) {
      throw std::runtime_error("invalid thermal field coupling at site " + std::to_string(i));
    }
  }
  return coupling;
}

jams::LorentzianBathParameters jams::LorentzianBathParameters::from_settings(
    const libconfig::Setting& settings) {
  const double omega0 = kTwoPi * jams::config_required<double>(settings, "lorentzian_omega0");
  const double linewidth = kTwoPi * jams::config_required<double>(settings, "lorentzian_gamma");
  if (!std::isfinite(omega0) || !std::isfinite(linewidth) || omega0 <= 0.0 || linewidth <= 0.0) {
    throw jams::ConfigException(settings, "Lorentzian frequency and linewidth must be finite and positive");
  }
  return {omega0, linewidth};
}

double jams::LorentzianBathParameters::amplitude(const double coupling) const {
  return coupling * pow4(omega0) / linewidth;
}

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

Thermostat* Thermostat::create(
    const std::string &thermostat_name, const double timestep,
    const std::span<const double> coupling, const std::optional<double> stage_interval,
    const std::optional<jams::LorentzianBathParameters> memory_bath) {
  std::cout << thermostat_name << " thermostat\n";

  const auto temperature = jams::config_required<double>(
      globals::config->lookup("physics"), "temperature");
  const auto name = capitalize(thermostat_name);
  // Preserve the timestep precision of existing factory callers.
  const auto legacy_timestep = static_cast<jams::Real>(timestep);
  if (coupling.size() != static_cast<std::size_t>(globals::num_spins)) {
    throw std::runtime_error("thermostat coupling must be specified for every site");
  }

  #if HAS_CUDA
  const double coloured_interval = stage_interval.value_or(legacy_timestep);
  if (name == "CLASSICAL-GPU" || name == "LANGEVIN-WHITE-GPU" || name == "CUDA_LANGEVIN_WHITE") {
    return new CudaThermostatClassical(temperature, coupling, legacy_timestep, globals::num_spins);
  }
  if (name == "QUANTUM-SPDE-GPU" || name == "LANGEVIN-BOSE-GPU" || name == "CUDA_LANGEVIN_COTH") {
    return new CudaThermostatQuantumSpde(temperature, coupling, coloured_interval, globals::num_spins);
  }
  if (name == "GENERAL-FFT-GPU" || name == "LANGEVIN-LORENTZIAN-GPU"
      || name == "LANGEVIN-ARBITRARY-GPU" || name == "CUDA_LANGEVIN_ARBITRARY") {
    const auto& settings = globals::config->lookup("thermostat");
    const auto spectrum = lowercase(memory_bath
        ? jams::config_optional<std::string>(settings, "spectrum", "classical-lorentzian")
        : jams::config_required<std::string>(settings, "spectrum"));
    if (stage_interval && (spectrum == "classical" || spectrum == "classical-ohmic")) {
      // Flat FFT noise is white. RS2 needs Wiener increments, not independent
      // stage samples that are combined with the deterministic RK weights.
      return new CudaThermostatClassical(temperature, coupling, legacy_timestep, globals::num_spins);
    }
    return new CudaThermostatGeneralFFT(
        temperature, coupling, coloured_interval, globals::num_spins, spectrum, memory_bath);
  }
  #else
  (void)memory_bath;
  #endif

  if (stage_interval) {
    throw std::runtime_error("thermostat " + thermostat_name + " is unsupported with a stage sampling interval");
  }

  if (name == "CLASSICAL-CPU" || name == "LANGEVIN-WHITE-CPU") {
    return new ThermostatClassical(temperature, coupling, legacy_timestep, globals::num_spins);
  }
  if (name == "QUANTUM-SPDE-CPU" || name == "LANGEVIN-BOSE-CPU" || name == "CPU_LANGEVIN_COTH") {
    return new ThermostatQuantumSpde(temperature, coupling, legacy_timestep, globals::num_spins);
  }

  #if HAS_CUDA
  if (name == "LANGEVIN-BOSE-SRK4-GPU") {
    return new jams::BoseEinsteinCudaSRK4Thermostat(temperature, coupling, legacy_timestep, globals::num_spins);
  }
  #endif

  throw std::runtime_error("unknown thermostat " + thermostat_name);
}
