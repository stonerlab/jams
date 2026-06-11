// Copyright 2014 Joseph Barker. All rights reserved.

#include <cinttypes>

#include <cmath>
#include <string>
#include <iomanip>

#include "jams/thermostats/cuda_thermostat_classical.h"
#include <jams/common.h>
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_common.h"

#include "jams/monitors/magnetisation.h"

CudaThermostatClassical::CudaThermostatClassical(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins)
: Thermostat(temperature, sigma, timestep, num_spins),
  sigma_spin_(num_spins) {
  std::cout << "\n  initialising classical-gpu thermostat\n";

  for(int i = 0; i < num_spins; ++i) {
    sigma_spin_(i) = static_cast<jams::Real>(sqrt((2.0 * kBoltzmannIU * globals::alpha(i)) /
                          (globals::mus(i) * globals::gyro(i) * timestep)));
    for (int j = 0; j < 3; ++j) {
      sigma_(i, j) = sigma_spin_(i);
    }
  }

  if (has_per_spin_temperature()) {
    sigma_sqrt_temperature_.resize(num_spins);
    const auto& sqrt_temperature = temperature_profile().sqrt_temperature();
    for (int i = 0; i < num_spins; ++i) {
      sigma_sqrt_temperature_(i) = sigma_spin_(i) * sqrt_temperature(i);
    }
  }

  std::cout << "  done\n\n";
}

void CudaThermostatClassical::update() {
  if (has_uniform_temperature() && this->temperature() == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise_.mutable_device_data(), 0, noise_.bytes(),jams::instance().cuda_master_stream().get()));
    return;
  }

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), jams::instance().cuda_master_stream().get()));
#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(jams::instance().curand_generator(), noise_.mutable_device_data(), (globals::num_spins3+(globals::num_spins3%2)), 0.0, 1.0));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), noise_.mutable_device_data(), (globals::num_spins3+(globals::num_spins3%2)), 0.0, 1.0));
#endif

  if (has_per_spin_temperature()) {
    cuda_array_elementwise_scale(globals::num_spins, 3, sigma_sqrt_temperature_.device_data(), 1.0, noise_.mutable_device_data(), 1, noise_.mutable_device_data(), 1, jams::instance().cuda_master_stream().get());
  } else {
    cuda_array_elementwise_scale(globals::num_spins, 3, sigma_spin_.device_data(), sqrt(this->temperature()), noise_.mutable_device_data(), 1, noise_.mutable_device_data(), 1, jams::instance().cuda_master_stream().get());
  }
}
