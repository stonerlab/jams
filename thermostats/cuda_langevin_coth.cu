// Copyright 2014 Joseph Barker. All rights reserved.

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cmath>
#include <string>
#include <iomanip>

#include "thermostats/cuda_langevin_coth.h"
#include "thermostats/cuda_langevin_coth_kernel.h"

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/magnetisation.h"

CudaLangevinCothThermostat::CudaLangevinCothThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  debug_(false),
  dev_noise_(3*num_spins) {
  ::output.write("\n  initialising CUDA Langevin semi-quantum noise thermostat\n");

  ::output.write("    initialising CUDA streams\n");

  if (cudaStreamCreate(&dev_stream_) != cudaSuccess){
    jams_error("Failed to create CUDA stream in CudaLangevinCothThermostat");
  }

  ::output.write("    initialising CURAND\n");

  // initialize and seed the CURAND generator on the device
  if (curandCreateGenerator(&dev_rng_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to create CURAND generator in CudaLangevinCothThermostat");
  }

  const uint64_t dev_rng_seed = rng.uniform()*18446744073709551615ULL;
  ::output.write("    seeding CURAND (%" PRIu64 ")", dev_rng_seed);

  if (curandSetPseudoRandomGeneratorSeed(dev_rng_, dev_rng_seed) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to set CURAND seed in CudaLangevinCothThermostat");
  }

  if (curandGenerateSeeds(dev_rng_) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to generate CURAND seeds in CudaLangevinCothThermostat");
  }

  ::output.write("    allocating GPU memory\n");
  dev_eta_.resize(6*globals::num_spins3);
  dev_zeta_.resize(8*globals::num_spins3);

  // initialize zeta and eta with random variables
  curandSetStream(dev_rng_, dev_stream_);
  if (curandGenerateNormalDouble(dev_rng_, dev_eta_.data(), dev_eta_.size(), 0.0, sqrt(this->temperature()))
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinCothThermostat::constructor");
  }

  if (curandGenerateNormalDouble(dev_rng_, dev_zeta_.data(), dev_zeta_.size(), 0.0, 0.0)
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinCothThermostat::constructor");
  }

}

void CudaLangevinCothThermostat::update() {
  int block_size = 64;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  // if (curandGenerateNormalDouble(dev_rng_, dev_eta_.data(), dev_eta_.size(), 0.0, sqrt(this->temperature())) != CURAND_STATUS_SUCCESS) {
  //   jams_error("curandGenerateNormalDouble failure in CudaLangevinCothThermostat::update");
  // }
  curandGenerateNormalDouble(dev_rng_, dev_eta_.data(), dev_eta_.size(), 0.0, sqrt(this->temperature()));
  coth_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>> (dev_noise_.data(), dev_zeta_.data(), dev_eta_.data(), 0.0005, globals::num_spins3);
}

CudaLangevinCothThermostat::~CudaLangevinCothThermostat() {
  curandDestroyGenerator(dev_rng_);
  cudaStreamDestroy(dev_stream_);
}
