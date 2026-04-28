// Copyright 2014 Joseph Barker. All rights reserved.

#include <iostream>
#include <memory>
#include <string>

#include "jams/helpers/utils.h"

#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include <jams/helpers/exception.h>

CudaThermostatQuantumSpde::CudaThermostatQuantumSpde(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins)
: Thermostat(temperature, sigma, timestep, num_spins)
{
  std::cout << "\n  initialising quantum-spde-gpu thermostat\n";

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

  bool use_gilbert_prefactor = jams::config_optional<bool>(
      globals::config->lookup("solver"), "gilbert_prefactor", false);
  std::cout << "    llg gilbert_prefactor " << use_gilbert_prefactor << "\n";

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      double denominator = 1.0;
      if (use_gilbert_prefactor) {
        denominator = 1.0 + pow2(globals::alpha(i));
      }
      sigma_(i,j) = static_cast<jams::Real>((kBoltzmannIU) * sqrt((2.0 * globals::alpha(i))
                                          / (kHBarIU * globals::gyro(i) * globals::mus(i) * denominator)));
    }
  }

  noise_generator_ = std::make_unique<jams::CudaQuantumSpdeNoiseGenerator>(
      num_spins * 3, delta_tau, omega_max, do_zero_point, cuda_stream_);

  if (initialization == "stationary") {
    std::cout << "initialising thermostat from stationary distribution @ ";
    std::cout << this->temperature() << "K" << std::endl;
    noise_generator_->initialize_stationary(this->temperature());
  } else {
    noise_generator_->initialize(jams::CudaQuantumSpdeNoiseGenerator::Initialization::Zero,
                                 this->temperature());
  }

  auto num_warm_up_steps = static_cast<unsigned>(t_warmup / dt_thermostat);
  if (do_warmup && num_warm_up_steps > 0) {
    std::cout << "warming up thermostat " << num_warm_up_steps << " steps @ ";
    std::cout << this->temperature() << "K" << std::endl;
    noise_generator_->warmup(num_warm_up_steps, this->temperature(),
                             noise_.mutable_device_data(), sigma_.device_data());
  }
}

void CudaThermostatQuantumSpde::update() {
  noise_generator_->update(noise_.mutable_device_data(), sigma_.device_data(), this->temperature());
}
