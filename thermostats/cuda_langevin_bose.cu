// Copyright 2014 Joseph Barker. All rights reserved.

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cmath>
#include <string>
#include <iomanip>

#include "thermostats/cuda_langevin_bose.h"
#include "thermostats/cuda_langevin_bose_kernel.h"

#include "core/globals.h"
#include "core/lattice.h"
#include "core/consts.h"

#include "monitors/magnetisation.h"

CudaLangevinBoseThermostat::CudaLangevinBoseThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  debug_(false),
  dev_noise_(3*num_spins) {
  ::output.write("\n  initialising CUDA Langevin semi-quantum noise thermostat\n");

  debug_ = true;

  if (debug_) {
    ::output.write("    DEBUG ON\n");
    std::string name = seedname + "_noise.dat";
    outfile_.open(name.c_str());
  }

  w_max_ = 50*kTHz;

  const double dt = ::config.lookup("sim.t_step");
  tau_ = (dt * kBoltzmann) / kHBar;

  ::output.write("    omega_max = %6.6f (THz)\n", w_max_ / kTHz);
  ::output.write("    hbar*w/kB = %4.4e\n", (kHBar * w_max_) / (kBoltzmann));
  ::output.write("    delta tau = %4.4e * T\n", tau_);

  ::output.write("    initialising CUDA streams\n");

  if (cudaStreamCreate(&dev_stream_) != cudaSuccess){
    jams_error("Failed to create CUDA stream in CudaLangevinBoseThermostat");
  }

  ::output.write("    initialising CURAND\n");

  // initialize and seed the CURAND generator on the device
  if (curandCreateGenerator(&dev_rng_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to create CURAND generator in CudaLangevinBoseThermostat");
  }

  const uint64_t dev_rng_seed = rng.uniform()*18446744073709551615ULL;
  ::output.write("    seeding CURAND (%" PRIu64 ")\n", dev_rng_seed);

  if (curandSetPseudoRandomGeneratorSeed(dev_rng_, dev_rng_seed) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to set CURAND seed in CudaLangevinBoseThermostat");
  }

  if (curandGenerateSeeds(dev_rng_) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to generate CURAND seeds in CudaLangevinBoseThermostat");
  }

  ::output.write("    allocating GPU memory\n");
  dev_eta_.resize(6*globals::num_spins3);
  dev_zeta_.resize(8*globals::num_spins3);

  // initialize zeta and eta with random variables
  curandSetStream(dev_rng_, dev_stream_);
  if (curandGenerateNormalDouble(dev_rng_, dev_eta_.data(), dev_eta_.size(), 0.0, 1.0)
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
  }

  if (curandGenerateNormalDouble(dev_rng_, dev_zeta_.data(), dev_zeta_.size(), 0.0, 0.0)
       != CURAND_STATUS_SUCCESS) {
    jams_error("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::constructor");
  }
}

void CudaLangevinBoseThermostat::update() {
  int block_size = 64;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  // if (curandGenerateNormalDouble(dev_rng_, dev_eta_.data(), dev_eta_.size(), 0.0, sqrt(this->temperature())) != CURAND_STATUS_SUCCESS) {
  //   jams_error("curandGenerateNormalDouble failure in CudaLangevinBoseThermostat::update");
  // }

  const double w_m = (kHBar * w_max_) / (kBoltzmann * this->temperature());
  // const double reduced_temperature = sqrt(this->temperature()) ;
  const double reduced_temperature = this->temperature() * sqrt( (2.0 * kBoltzmann * globals::alpha(0) * globals::mus(0)) / (kBohrMagneton) );


  curandGenerateNormalDouble(dev_rng_, dev_eta_.data(), dev_eta_.size(), 0.0, 1.0);
  bose_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>> (dev_noise_.data(), dev_zeta_.data(), dev_eta_.data(), tau_ * this->temperature(), reduced_temperature, w_m, globals::num_spins3);

  if (debug_) {
    jblib::Array<double, 1> dbg_noise(dev_noise_.size(), 0.0);
    dev_noise_.copy_to_host_array(dbg_noise);
    outfile_ << dbg_noise(0) << std::endl;
  }
}

CudaLangevinBoseThermostat::~CudaLangevinBoseThermostat() {
  curandDestroyGenerator(dev_rng_);
  cudaStreamDestroy(dev_stream_);
  if (debug_) {
    outfile_.close();
  }
}
