// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/cpu_thermostat_classical.h"

#include <cmath>
#include <iostream>
#include <random>

#include <jams/common.h>

CpuWhiteNoiseGenerator::CpuWhiteNoiseGenerator(const jams::Real& temperature,
                                               const jams::Real timestep,
                                               const int num_vectors)
    : NoiseGenerator(temperature, num_vectors) {
  static_cast<void>(timestep);
  std::cout << "\n  initialising classical-cpu noise generator\n";
}

void CpuWhiteNoiseGenerator::update() {
  if (this->temperature() == 0) {
    zero(noise_);
    return;
  }

  std::normal_distribution<double> normal_distribution(
      0.0, std::sqrt(static_cast<double>(this->temperature())));
  auto& random_generator = jams::instance().random_generator();
  for (auto i = 0; i < noise_.size(0); ++i) {
    for (auto j = 0; j < noise_.size(1); ++j) {
      noise_(i, j) = static_cast<jams::Real>(normal_distribution(random_generator));
    }
  }
}
