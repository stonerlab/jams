// Copyright 2014 Joseph Barker. All rights reserved.

#include <cinttypes>

#include <cmath>
#include <string>
#include <iomanip>

#include "cuda_langevin_white.h"

#include "jams/cuda/cuda_array_kernels.h"
#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/core/solver.h"

#include "jams/monitors/magnetisation.h"

using namespace std;

CudaLangevinWhiteThermostat::CudaLangevinWhiteThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  is_synchronised_(false),
  dev_noise_((3*num_spins+((3*num_spins)%2))),
  dev_sigma_(num_spins),
  dev_rng_(nullptr),
  dev_stream_(nullptr) {
  cout << "\n  initialising CUDA Langevin white noise thermostat\n";
  cout << "    initialising CURAND\n";

  // initialize and seed the CURAND generator on the device
  if (curandCreateGenerator(&dev_rng_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to create CURAND generator in CudaLangevinWhiteThermostat");
  }

  cout << "    creating stream\n";
  cudaStreamCreate(&dev_stream_);
  curandSetStream(dev_rng_, dev_stream_);
  
  auto dev_rng_seed = static_cast<uint64_t>(std::random_device()());
  cout << "    seeding CURAND " << dev_rng_seed << "\n";
  if (curandSetPseudoRandomGeneratorSeed(dev_rng_, dev_rng_seed) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to set CURAND seed in CudaLangevinWhiteThermostat");
  }

  cout << "    generating seeds\n";
  if (curandGenerateSeeds(dev_rng_) != CURAND_STATUS_SUCCESS) {
    jams_error("Failed to generate CURAND seeds in CudaLangevinWhiteThermostat");
  }

  bool use_gilbert_prefactor = jams::config_optional<bool>(config->lookup("solver"), "gilbert_prefactor", false);
  cout << "    llg gilbert_prefactor " << use_gilbert_prefactor << "\n";


  for(int i = 0; i < num_spins; ++i) {
    double denominator = 1.0;
    if (use_gilbert_prefactor) {
      denominator = 1.0 + pow2(globals::alpha(i));
    }
    sigma_(i) = sqrt( (2.0 * kBoltzmann * globals::alpha(i) * globals::mus(i)) / (solver->time_step() * kGyromagneticRatio * kBohrMagneton * denominator) );
  }

  cout << "    transfering sigma to device\n";
  dev_sigma_ = jblib::CudaArray<double, 1>(sigma_);

  is_synchronised_ = false;

  cout << "  done\n\n";
}

void CudaLangevinWhiteThermostat::update() {
  curandGenerateNormalDouble(dev_rng_, dev_noise_.data(), (globals::num_spins3+(globals::num_spins3%2)), 0.0, 1.0);
  cuda_array_elementwise_scale(globals::num_spins, 3, dev_sigma_.data(), sqrt(this->temperature()), dev_noise_.data(), 1, dev_noise_.data(), 1, dev_stream_);
  is_synchronised_ = false;
}

CudaLangevinWhiteThermostat::~CudaLangevinWhiteThermostat() {
  curandDestroyGenerator(dev_rng_);
  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }
}
