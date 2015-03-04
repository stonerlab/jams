// Copyright 2014 Joseph Barker. All rights reserved.

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cmath>
#include <string>
#include <iomanip>

#include "thermostats/cuda_langevin_white.h"

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation.h"

CudaLangevinWhiteThermostat::CudaLangevinWhiteThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  dev_noise_(3*num_spins) {

  ::output.write("\ninitialising CUDA Langevin white noise thermostat...\n");

  ::output.write("  initialising CURAND...\n");

  // initialize and seed the CURAND generator on the device
  if (curandCreateGenerator(&dev_rng_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to create CURAND generator in CudaLangevinWhiteThermostat");
  }

  const uint64_t dev_rng_seed = rng.uniform()*18446744073709551615ULL;
  ::output.write("    seeding CURAND (%" PRIu64 ")", dev_rng_seed);

  if (curandSetPseudoRandomGeneratorSeed(dev_rng_, dev_rng_seed) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to set CURAND seed in CudaLangevinWhiteThermostat");
  }

  if (curandGenerateSeeds(dev_rng_) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to generate CURAND seeds in CudaLangevinWhiteThermostat");
  }
}

void CudaLangevinWhiteThermostat::update() {
  if (curandGenerateNormalDouble(dev_rng_, dev_noise_.data(), (globals::num_spins3+(globals::num_spins3%2)), 0.0, sqrt(this->temperature()))
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinWhiteThermostat::update");
  }
}

CudaLangevinWhiteThermostat::~CudaLangevinWhiteThermostat() {
  curandDestroyGenerator(dev_rng_);
}
