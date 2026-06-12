// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/thermostat_classical.h"

#include <cmath>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"

#if HAS_OMP
#include <omp.h>
#endif

ThermostatClassical::ThermostatClassical(const jams::Real& temperature,
                                         const jams::Real& sigma,
                                         const jams::Real timestep,
                                         const int num_spins)
    : Thermostat(temperature, sigma, timestep, num_spins),
      sigma_spin_(num_spins),
      random_generator_(pcg_extras::seed_seq_from<pcg32>(
          jams::instance().random_generator()())) {
  ensure_random_generators(1);

  auto sigma_spin_view = sigma_spin_.mutable_host_view();
  auto alpha_view = globals::alpha.host_view();
  auto mu_view = globals::mus.host_view();
  auto gyro_view = globals::gyro.host_view();
  auto* sigma_spin_values = sigma_spin_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* mu_values = mu_view.data();
  const auto* gyro_values = gyro_view.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < num_spins; ++i) {
    sigma_spin_values[i] = static_cast<jams::Real>(
        std::sqrt((2.0 * kBoltzmannIU * alpha_values[i]) /
                  (mu_values[i] * gyro_values[i] * timestep)));
  }
}

void ThermostatClassical::ensure_random_generators(const int count) {
  if (random_generators_.empty() && count > 0) {
    random_generators_.push_back(random_generator_);
  }
  while (static_cast<int>(random_generators_.size()) < count) {
    random_generators_.emplace_back(
        pcg_extras::seed_seq_from<pcg32>(random_generator_()));
  }
}

void ThermostatClassical::update() {
  if (has_uniform_temperature() && temperature() == 0.0) {
    noise_.zero();
    return;
  }

  int num_generators = 1;
#if HAS_OMP
  num_generators = omp_get_max_threads();
#endif
  ensure_random_generators(num_generators);

  auto noise_view = noise_.mutable_host_view();
  const auto sigma_spin_view = sigma_spin_.host_view();
  auto* noise_values = noise_view.data();
  const auto* sigma_spin_values = sigma_spin_view.data();

  std::normal_distribution<> normal_distribution;
  if (has_uniform_temperature()) {
    const auto sqrt_temperature = std::sqrt(static_cast<double>(temperature()));
#if HAS_OMP
#pragma omp parallel
    {
      const auto thread_id = omp_get_thread_num();
      auto& generator = random_generators_[thread_id];
      std::normal_distribution<> thread_distribution;
#pragma omp for schedule(static)
      for (auto i = 0; i < globals::num_spins; ++i) {
        const auto scale = static_cast<jams::Real>(sigma_spin_values[i] * sqrt_temperature);
        const auto offset = 3 * i;
        noise_values[offset] = static_cast<jams::Real>(thread_distribution(generator)) * scale;
        noise_values[offset + 1] = static_cast<jams::Real>(thread_distribution(generator)) * scale;
        noise_values[offset + 2] = static_cast<jams::Real>(thread_distribution(generator)) * scale;
      }
    }
#else
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto scale = static_cast<jams::Real>(sigma_spin_values[i] * sqrt_temperature);
      const auto offset = 3 * i;
      noise_values[offset] = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
      noise_values[offset + 1] = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
      noise_values[offset + 2] = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
    }
#endif
    return;
  }

  const auto& sqrt_temperature = temperature_profile().sqrt_temperature();
  const auto sqrt_temperature_view = sqrt_temperature.host_view();
  const auto* sqrt_temperature_values = sqrt_temperature_view.data();
#if HAS_OMP
#pragma omp parallel
  {
    const auto thread_id = omp_get_thread_num();
    auto& generator = random_generators_[thread_id];
    std::normal_distribution<> thread_distribution;
#pragma omp for schedule(static)
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto scale = sigma_spin_values[i] * sqrt_temperature_values[i];
      const auto offset = 3 * i;
      noise_values[offset] = static_cast<jams::Real>(thread_distribution(generator)) * scale;
      noise_values[offset + 1] = static_cast<jams::Real>(thread_distribution(generator)) * scale;
      noise_values[offset + 2] = static_cast<jams::Real>(thread_distribution(generator)) * scale;
    }
  }
#else
  for (auto i = 0; i < globals::num_spins; ++i) {
    const auto scale = sigma_spin_values[i] * sqrt_temperature_values[i];
    const auto offset = 3 * i;
    noise_values[offset] = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
    noise_values[offset + 1] = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
    noise_values[offset + 2] = static_cast<jams::Real>(normal_distribution(random_generator_)) * scale;
  }
#endif
}
