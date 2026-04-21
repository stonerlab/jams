// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/core/thermostat.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "jams/core/globals.h"
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/utils.h"
#include "jams/interface/config.h"

#include "jams/thermostats/cuda_thermostat_classical.h"
#include "jams/thermostats/cuda_thermostat_general_fft.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde.h"
#include "jams/thermostats/thm_bose_einstein_cuda_srk4.h"

namespace {

const libconfig::Setting* thermostat_settings() {
  if (globals::config && globals::config->exists("thermostat")) {
    return &globals::config->lookup("thermostat");
  }
  return nullptr;
}

jams::MultiArray<jams::Real, 1> make_uniform_sigma(const jams::Real value) {
  jams::MultiArray<jams::Real, 1> sigma(globals::num_spins);
  std::fill(sigma.begin(), sigma.end(), value);
  return sigma;
}

jams::MultiArray<jams::Real, 1> make_classical_sigma(const jams::Real timestep) {
  const bool use_gilbert_prefactor = jams::config_optional<bool>(
      globals::config->lookup("solver"), "gilbert_prefactor", false);

  jams::MultiArray<jams::Real, 1> sigma(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    double denominator = 1.0;
    if (use_gilbert_prefactor) {
      denominator = 1.0 + pow2(globals::alpha(i));
    }
    sigma(i) = static_cast<jams::Real>(std::sqrt(
        (2.0 * kBoltzmannIU * globals::alpha(i))
        / (globals::mus(i) * globals::gyro(i) * timestep * denominator)));
  }
  return sigma;
}

jams::MultiArray<jams::Real, 1> make_quantum_sigma() {
  jams::MultiArray<jams::Real, 1> sigma(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    sigma(i) = static_cast<jams::Real>(
        kBoltzmannIU
        * std::sqrt((2.0 * globals::alpha(i))
                    / (kHBarIU * globals::gyro(i) * globals::mus(i))));
  }
  return sigma;
}

jams::MultiArray<jams::Real, 1> make_general_fft_sigma() {
  jams::MultiArray<jams::Real, 1> sigma(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i) {
    sigma(i) = static_cast<jams::Real>(
        std::sqrt(globals::alpha(i) / (globals::mus(i) * globals::gyro(i))));
  }
  return sigma;
}

}  // namespace

Thermostat::Thermostat(std::unique_ptr<NoiseGenerator> generator,
                       jams::MultiArray<jams::Real, 1> sigma)
    : temperature_(generator ? generator->temperature() : jams::Real(0.0)),
      generator_(std::move(generator)),
      sigma_(std::move(sigma)),
      noise_(generator_ ? generator_->num_vectors() : 0, 3) {
  noise_.zero();

#if HAS_CUDA
  cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
#endif
}

Thermostat::~Thermostat() {
#if HAS_CUDA
  if (done_ != nullptr) {
    cudaEventDestroy(done_);
    done_ = nullptr;
  }
#endif
}

void Thermostat::update() {
  generator_->set_temperature(temperature_);
  generator_->update();

#if HAS_CUDA
  generator_->record_done();
  generator_->wait_on(cuda_stream_.get());

  cuda_array_elementwise_scale(
      generator_->num_vectors(),
      3,
      sigma_.device_data(),
      jams::Real(1.0),
      const_cast<jams::Real*>(generator_->device_data()),
      1,
      noise_.device_data(),
      1,
      cuda_stream_.get());
#else
  for (auto i = 0; i < generator_->num_vectors(); ++i) {
    for (int j = 0; j < 3; ++j) {
      noise_(i, j) = sigma_(i) * generator_->field(i, j);
    }
  }
#endif
}

Thermostat* Thermostat::create(const std::string& thermostat_name, const jams::Real timestep) {
  std::cout << thermostat_name << " thermostat\n";

  const auto temperature = jams::config_required<double>(
      globals::config->lookup("physics"), "temperature");

#if HAS_CUDA
  const auto* settings = thermostat_settings();

  if (capitalize(thermostat_name) == "CLASSICAL-GPU"
      || capitalize(thermostat_name) == "LANGEVIN-WHITE-GPU"
      || capitalize(thermostat_name) == "CUDA_LANGEVIN_WHITE") {
    return new Thermostat(
        std::make_unique<CudaWhiteNoiseGenerator>(temperature, timestep, globals::num_spins),
        make_classical_sigma(timestep));
  }

  if (capitalize(thermostat_name) == "QUANTUM-SPDE-GPU"
      || capitalize(thermostat_name) == "LANGEVIN-BOSE-GPU"
      || capitalize(thermostat_name) == "CUDA_LANGEVIN_COTH") {
    CudaQuantumSpdeNoiseGeneratorConfig config;
    if (settings != nullptr) {
      settings->lookupValue("zero_point", config.zero_point);
      settings->lookupValue("w_max", config.omega_max);
    }
    return new Thermostat(
        std::make_unique<CudaQuantumSpdeNoiseGenerator>(
            temperature, timestep, globals::num_spins, config),
        make_quantum_sigma());
  }

  if (capitalize(thermostat_name) == "LANGEVIN-BOSE-SRK4-GPU") {
    BoseEinsteinCudaSRK4NoiseGeneratorConfig config;
    if (settings != nullptr) {
      settings->lookupValue("warmup_time", config.warmup_time_ps);
    }
    return new Thermostat(
        std::make_unique<jams::BoseEinsteinCudaSRK4NoiseGenerator>(
            temperature, timestep, globals::num_spins, config),
        make_quantum_sigma());
  }

  if (capitalize(thermostat_name) == "GENERAL-FFT-GPU"
      || capitalize(thermostat_name) == "LANGEVIN-LORENTZIAN-GPU"
      || capitalize(thermostat_name) == "LANGEVIN-ARBITRARY-GPU"
      || capitalize(thermostat_name) == "CUDA_LANGEVIN_ARBITRARY") {
    CudaGeneralFFTNoiseGeneratorConfig config;
    config.write_diagnostics = true;
    if (settings != nullptr) {
      settings->lookupValue("spectrum", config.spectrum);
      settings->lookupValue("lorentzian_omega0", config.lorentzian_omega0);
      settings->lookupValue("lorentzian_gamma", config.lorentzian_gamma);
      settings->lookupValue("write_diagnostics", config.write_diagnostics);
      settings->lookupValue("write_debug_output", config.write_diagnostics);
    }
    return new Thermostat(
        std::make_unique<CudaGeneralFFTNoiseGenerator>(
            temperature, timestep, globals::num_spins, config),
        make_general_fft_sigma());
  }
#endif

  throw std::runtime_error("unknown thermostat " + thermostat_name);
}
