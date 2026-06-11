// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/thermostat_classical.h"

#include <cmath>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"

ThermostatClassical::ThermostatClassical(const jams::Real& temperature,
                                         const jams::Real& sigma,
                                         const jams::Real timestep,
                                         const int num_spins)
    : Thermostat(temperature, sigma, timestep, num_spins),
      sigma_spin_(num_spins),
      random_generator_(pcg_extras::seed_seq_from<pcg32>(
          jams::instance().random_generator()())) {
  for (int i = 0; i < num_spins; ++i) {
    sigma_spin_(i) = static_cast<jams::Real>(
        std::sqrt((2.0 * kBoltzmannIU * globals::alpha(i)) /
                  (globals::mus(i) * globals::gyro(i) * timestep)));
  }
}

void ThermostatClassical::update() {
  if (has_uniform_temperature() && temperature() == 0.0) {
    noise_.zero();
    return;
  }

  std::normal_distribution<> normal_distribution;
  if (has_uniform_temperature()) {
    const auto sqrt_temperature = std::sqrt(static_cast<double>(temperature()));
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto scale = static_cast<jams::Real>(sigma_spin_(i) * sqrt_temperature);
      for (auto j = 0; j < 3; ++j) {
        noise_(i, j) = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
      }
    }
    return;
  }

  const auto& sqrt_temperature = temperature_profile().sqrt_temperature();
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto scale = sigma_spin_(i) * sqrt_temperature(i);
    for (auto j = 0; j < 3; ++j) {
      noise_(i, j) = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
    }
  }
}
