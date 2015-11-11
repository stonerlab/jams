// Copyright 2014 Joseph Barker. All rights reserved.

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cmath>
#include <string>
#include <iomanip>

#include "thermostats/cuda_langevin_white.h"

#include "core/cuda_array_kernels.h"
#include "core/consts.h"
#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation.h"

CudaLangevinWhiteThermostat::CudaLangevinWhiteThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  is_synchronised_(false),
  dev_noise_((3*num_spins+((3*num_spins)%2)))
    {
  ::output.write("\n  initialising CUDA Langevin white noise thermostat\n");

  ::output.write("    initialising CURAND\n");

  // initialize and seed the CURAND generator on the device
  if (curandCreateGenerator(&dev_rng_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to create CURAND generator in CudaLangevinWhiteThermostat");
  }

  const uint64_t dev_rng_seed = rng.uniform()*18446744073709551615ULL;
  ::output.write("    seeding CURAND (%" PRIu64 ")", dev_rng_seed);

  // put into the same stream as the integration kernels so that we don't mix the new
  // and old numbers while the integration kernels are running
  // curandSetStream(dev_rng_, ::cuda_streams[0]);

  if (curandSetPseudoRandomGeneratorSeed(dev_rng_, dev_rng_seed) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to set CURAND seed in CudaLangevinWhiteThermostat");
  }

  if (curandGenerateSeeds(dev_rng_) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to generate CURAND seeds in CudaLangevinWhiteThermostat");
  }
  // sigma.resize(num_spins);
  for(int i = 0; i < num_spins; ++i) {
    sigma_(i) = sqrt( (2.0 * kBoltzmann * globals::alpha(i) * globals::mus(i)) / (solver->time_step() * kBohrMagneton) );
  }

  dev_sigma_ = jblib::CudaArray<double, 1>(sigma_);

  is_synchronised_ = false;
}

void CudaLangevinWhiteThermostat::update() {
  curandGenerateNormalDouble(dev_rng_, dev_noise_.data(), (globals::num_spins3+(globals::num_spins3%2)), 0.0, 1.0);
  cuda_array_elementwise_scale(globals::num_spins, 3, dev_sigma_.data(), sqrt(this->temperature()), dev_noise_.data(), 1, dev_noise_.data(), 1);
  is_synchronised_ = false;
}

CudaLangevinWhiteThermostat::~CudaLangevinWhiteThermostat() {
  curandDestroyGenerator(dev_rng_);
}
