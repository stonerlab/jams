// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "jams/helpers/utils.h"

#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/finite.h"
#include <jams/helpers/exception.h>

namespace {

int checked_spin_count_for_cuda_quantum_spde(const int num_spins) {
  (void)jams::quantum_spde_cuda_process_count(num_spins);
  return num_spins;
}

}  // namespace

CudaThermostatQuantumSpde::CudaThermostatQuantumSpde(
    const jams::Real &temperature, const jams::Real &sigma, const double timestep,
    const int num_spins)
: Thermostat(temperature, sigma, static_cast<jams::Real>(timestep),
             checked_spin_count_for_cuda_quantum_spde(num_spins))
{
  std::cout << "\n  initialising quantum-spde-gpu thermostat\n";

  if (!jams::is_finite(timestep) || timestep <= 0.0) {
    throw std::runtime_error("quantum-spde-gpu requires a positive finite sampling interval");
  }

  cuda_stream_ = CudaStream(CudaStream::Priority::LOW);
  const libconfig::Setting* thermostat_settings = nullptr;
  if (globals::config->exists("thermostat")) {
    thermostat_settings = &globals::config->lookup("thermostat");
  }

  const bool do_zero_point = thermostat_settings != nullptr
      && jams::config_optional<bool>(*thermostat_settings, "zero_point", false);

  double t_warmup = 1e-10; // 0.1 ns
  if (thermostat_settings != nullptr) {
    t_warmup = jams::config_optional<double>(*thermostat_settings, "warmup_time", t_warmup);
  }
  t_warmup = t_warmup / 1e-12; // convert to ps
  const bool do_warmup = thermostat_settings != nullptr
      && jams::config_optional<bool>(*thermostat_settings, "warmup", false);
  const auto initialization = lowercase(thermostat_settings != nullptr
      ? jams::config_optional<std::string>(*thermostat_settings, "initialization", "stationary")
      : std::string("stationary"));
  if (initialization != "stationary" && initialization != "zero") {
    throw jams::ConfigException(
        *thermostat_settings, "initialization must be either 'stationary' or 'zero'");
  }

  double omega_max = 25.0 * kTwoPi;
  if (thermostat_settings != nullptr) {
    omega_max = jams::config_optional<double>(*thermostat_settings, "w_max", omega_max);
  }

  double dt_thermostat = timestep;
  const double delta_tau = (dt_thermostat * kBoltzmannIU) / kHBarIU;

  std::cout << "    omega_max (THz) " << omega_max / (kTwoPi) << "\n";
  std::cout << "    hbar*w/kB " << (kHBarIU * omega_max) / (kBoltzmannIU) << "\n";
  std::cout << "    t_step " << dt_thermostat << "\n";
  std::cout << "    delta tau " << delta_tau << "\n";
  std::cout << "    initialization " << initialization << "\n";
  std::cout << "    warmup " << std::boolalpha << do_warmup << "\n";
  std::cout << "    zero_point " << do_zero_point << "\n";

  const int process_count = jams::quantum_spde_cuda_process_count(num_spins);

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      sigma_(i,j) = static_cast<jams::Real>((kBoltzmannIU) * sqrt((2.0 * globals::alpha(i))
                                          * globals::inv_mus(i) / (kHBarIU * globals::gyro(i))));
    }
  }

  noise_generator_ = std::make_unique<jams::CudaQuantumSpdeNoiseGenerator>(
      process_count, delta_tau, omega_max, do_zero_point, cuda_stream_);

  if (has_per_spin_temperature()) {
    process_temperature_.resize(process_count);
    const auto& spin_temperature = temperature_profile().temperature();
    for (int i = 0; i < num_spins; ++i) {
      for (int j = 0; j < 3; ++j) {
        process_temperature_(3 * i + j) = spin_temperature(i);
      }
    }
  }

  if (initialization == "stationary") {
    if (has_per_spin_temperature()) {
      std::cout << "initialising thermostat from per-spin stationary distribution" << std::endl;
      noise_generator_->initialize_stationary(process_temperature_);
    } else {
      std::cout << "initialising thermostat from stationary distribution @ ";
      std::cout << this->temperature() << "K" << std::endl;
      noise_generator_->initialize_stationary(this->temperature());
    }
  } else {
    initial_sample_ready_ = true;  // The initial state and the output buffer are zero.
    if (has_per_spin_temperature()) {
      noise_generator_->initialize(jams::CudaQuantumSpdeNoiseGenerator::Initialization::Zero,
                                   process_temperature_);
    } else {
      noise_generator_->initialize(jams::CudaQuantumSpdeNoiseGenerator::Initialization::Zero,
                                   this->temperature());
    }
  }

  const double warm_up_steps = do_warmup ? std::floor(t_warmup / dt_thermostat) : 0.0;
  if (!jams::is_finite(warm_up_steps) || warm_up_steps < 0.0 ||
      warm_up_steps > std::numeric_limits<unsigned>::max()) {
    throw jams::ConfigException(*thermostat_settings, "invalid SPDE warmup duration");
  }
  const auto num_warm_up_steps = static_cast<unsigned>(warm_up_steps);
  if (num_warm_up_steps > 0) {
    if (has_per_spin_temperature()) {
      std::cout << "warming up thermostat " << num_warm_up_steps
                << " steps with per-spin temperatures" << std::endl;
      noise_generator_->warmup(num_warm_up_steps, process_temperature_,
                               noise_.mutable_device_data(), sigma_.device_data());
    } else {
      std::cout << "warming up thermostat " << num_warm_up_steps << " steps @ ";
      std::cout << this->temperature() << "K" << std::endl;
      noise_generator_->warmup(num_warm_up_steps, this->temperature(),
                               noise_.mutable_device_data(), sigma_.device_data());
    }
    initial_sample_ready_ = true;
  }
}

void CudaThermostatQuantumSpde::sample_initial() {
  if (initial_sample_ready_) return;

  // Anchor the first stationary sample at t=0. This also samples the
  // zero-point term, whose field depends on the interval's innovation.
  update();
  initial_sample_ready_ = true;
}

void CudaThermostatQuantumSpde::update() {
  if (has_per_spin_temperature()) {
    noise_generator_->update(noise_.mutable_device_data(), sigma_.device_data(),
                             process_temperature_);
    return;
  }

  noise_generator_->update(noise_.mutable_device_data(), sigma_.device_data(), this->temperature());
}
