// Copyright 2014 Joseph Barker. All rights reserved.

#include <cinttypes>

#include <cmath>
#include <string>
#include <iomanip>

#include "cuda_langevin_white.h"
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

CudaLangevinWhiteThermostat::CudaLangevinWhiteThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  dev_stream_(nullptr) {
  std::cout << "\n  initialising CUDA Langevin white noise thermostat\n";

  cudaStreamCreate(&dev_stream_);

  bool use_gilbert_prefactor = jams::config_optional<bool>(
      globals::config->lookup("solver"), "gilbert_prefactor", false);
  std::cout << "    llg gilbert_prefactor " << use_gilbert_prefactor << "\n";


  for(int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      double denominator = 1.0;
      if (use_gilbert_prefactor) {
        denominator = 1.0 + pow2(globals::alpha(i));
      }
      sigma_(i, j) = sqrt((2.0 * kBoltzmannIU * globals::alpha(i)) /
                          (globals::mus(i) * globals::gyro(i) * globals::solver->time_step() * denominator));
    }
  }
  std::cout << "  done\n\n";
}

void CudaLangevinWhiteThermostat::update() {
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_stream_));
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), noise_.device_data(), (globals::num_spins3+(globals::num_spins3%2)), 0.0, 1.0));
  cuda_array_elementwise_scale(globals::num_spins, 3, sigma_.device_data(), sqrt(this->temperature()), noise_.device_data(), 1, noise_.device_data(), 1, dev_stream_);
}