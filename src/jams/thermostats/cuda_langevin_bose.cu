// Copyright 2014 Joseph Barker. All rights reserved.

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cmath>
#include <string>
#include <iomanip>
#include "jams/core/cuda_array_kernels.h"

#include "jams/thermostats/cuda_langevin_bose.h"
#include "jams/thermostats/cuda_langevin_bose_kernel.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/consts.h"
#include "jams/core/output.h"
#include "jams/core/rand.h"
#include "jams/core/error.h"

#include "jams/monitors/magnetisation.h"

CudaLangevinBoseThermostat::CudaLangevinBoseThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  debug_(false),
  dev_noise_(3 * num_spins, 0.0),
  dev_zeta5_(num_spins * 3, 0.0),
  dev_zeta5p_(num_spins * 3, 0.0),
  dev_zeta6_(num_spins * 3, 0.0),
  dev_zeta6p_(num_spins * 3, 0.0),
  dev_eta0_(4 * num_spins * 3, 0.0),
  dev_eta1a_(2 * num_spins * 3, 0.0),
  dev_eta1b_(2 * num_spins * 3, 0.0),
  dev_sigma_(num_spins, 0.0)
 {
  ::output->write("\n  initialising CUDA Langevin semi-quantum noise thermostat\n");

  debug_ = false;

  if (debug_) {
    ::output->write("    DEBUG ON\n");
    std::string name = seedname + "_noise.dat";
    outfile_.open(name.c_str());
  }

  w_max_ = 100*kTHz;

  config->lookupValue("sim.w_max", w_max_);

  const double dt = ::config->lookup("sim.t_step");
  tau_ = (dt * kBoltzmann) / kHBar;

  ::output->write("    omega_max = %6.6f (THz)\n", w_max_ / kTHz);
  ::output->write("    hbar*w/kB = %4.4e\n", (kHBar * w_max_) / (kBoltzmann));
  ::output->write("    delta tau = %4.4e * T\n", tau_);

  ::output->write("    initialising CUDA streams\n");

  if (cudaStreamCreate(&dev_stream_) != cudaSuccess){
    jams_error("Failed to create CUDA stream in CudaLangevinBoseThermostat");
  }

  if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess){
    jams_error("Failed to create CURAND stream in CudaLangevinBoseThermostat");
  }

  ::output->write("    initialising CURAND\n");

  // initialize and seed the CURAND generator on the device
  if (curandCreateGenerator(&dev_rng_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to create CURAND generator in CudaLangevinBoseThermostat");
  }

  // initialize zeta and eta with random variables
  curandSetStream(dev_rng_, dev_curand_stream_);

  const uint64_t dev_rng_seed = rng->uniform()*18446744073709551615ULL;
  ::output->write("    seeding CURAND (%" PRIu64 ")\n", dev_rng_seed);

  if (curandSetPseudoRandomGeneratorSeed(dev_rng_, dev_rng_seed) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to set CURAND seed in CudaLangevinBoseThermostat");
  }

  if (curandGenerateSeeds(dev_rng_) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to generate CURAND seeds in CudaLangevinBoseThermostat");
  }

  ::output->write("    allocating GPU memory\n");

  if (curandGenerateNormalDouble(dev_rng_, dev_eta0_.data(), dev_eta0_.size(), 0.0, 1.0)
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
  }

  if (curandGenerateNormalDouble(dev_rng_, dev_eta1a_.data(), dev_eta1a_.size(), 0.0, 1.0)
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
  }

  if (curandGenerateNormalDouble(dev_rng_, dev_eta1b_.data(), dev_eta1b_.size(), 0.0, 1.0)
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
  }

  jblib::Array<double, 2> scale(num_spins, 3);
  for(int i = 0; i < num_spins; ++i) {
    for(int j = 0; j < 3; ++j) {
      scale(i, j) = (kBoltzmann) * sqrt( (2.0 * globals::alpha(i) * globals::mus(i)) / ( kHBar * kGyromagneticRatio * kBohrMagneton) );
    }
  }

  dev_sigma_ = jblib::CudaArray<double, 1>(scale);

  // const int num_warmup_steps = 10;
  const int num_warmup_steps = 1000000;

  ::output->write("    warming up thermostat (%8.2f ns @ %8.2f K)\n", ((dt *num_warmup_steps) / 1.0e-9), this->temperature());

  for (int i = 0; i < num_warmup_steps; ++i) {
    update();
  }
}

void CudaLangevinBoseThermostat::update() {
  int block_size = 96;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  swap(dev_eta1a_, dev_eta1b_);

  curandGenerateNormalDouble(dev_rng_, dev_eta1a_.data(), dev_eta1a_.size(), 0.0, 1.0);

  bose_coth_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>> (
    dev_noise_.data(),
    dev_zeta5_.data(),
    dev_zeta5p_.data(),
    dev_zeta6_.data(),
    dev_zeta6p_.data(),
    dev_eta1b_.data(),
    dev_sigma_.data(),
    tau_ * this->temperature(),
    this->temperature(),
    (kHBar * w_max_) / (kBoltzmann * this->temperature()),  // w_m
    globals::num_spins3);
}

CudaLangevinBoseThermostat::~CudaLangevinBoseThermostat() {
  curandDestroyGenerator(dev_rng_);
  cudaStreamDestroy(dev_stream_);
  cudaStreamDestroy(dev_curand_stream_);
  if (debug_) {
    outfile_.close();
  }
}
