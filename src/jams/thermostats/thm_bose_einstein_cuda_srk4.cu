// thm_bose_einstein_cuda_srk4.cu                                      -*-C++-*-

#include "jams/thermostats/thm_bose_einstein_cuda_srk4.h"

#include <cmath>
#include <iostream>

#include <jams/common.h>

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/thermostats/thm_bose_einstein_cuda_srk4_kernel.cuh"

jams::BoseEinsteinCudaSRK4NoiseGenerator::BoseEinsteinCudaSRK4NoiseGenerator(
    const double& temperature,
    const double timestep,
    const int num_vectors,
    const BoseEinsteinCudaSRK4NoiseGeneratorConfig& config)
    : NoiseGenerator(temperature, num_vectors),
      delta_tau_((timestep * kBoltzmannIU) / kHBarIU),
      num_channels_(3 * num_vectors) {
  std::cout << "\n  initialising CUDA Langevin semi-quantum noise generator\n";
  jams_warning("This thermostat is currently broken. Do not use for production work.");

  num_warm_up_steps_ = static_cast<unsigned>(config.warmup_time_ps / timestep);

  zero(w5_.resize(num_channels_));
  zero(v5_.resize(num_channels_));
  zero(w6_.resize(num_channels_));
  zero(v6_.resize(num_channels_));
  zero(psi5_.resize(2 * num_channels_));
  zero(psi6_.resize(2 * num_channels_));

  uint64_t dev_rng_seed = jams::instance().random_generator()();

  CHECK_CURAND_STATUS(curandCreateGenerator(&prng5_, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND_STATUS(curandSetStream(prng5_, dev_stream5_.get()));
  CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(prng5_, dev_rng_seed));
  CHECK_CURAND_STATUS(curandGenerateSeeds(prng5_));
  cudaEventCreate(&event5_);

  dev_rng_seed = jams::instance().random_generator()();
  CHECK_CURAND_STATUS(curandCreateGenerator(&prng6_, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND_STATUS(curandSetStream(prng6_, dev_stream6_.get()));
  CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(prng6_, dev_rng_seed));
  CHECK_CURAND_STATUS(curandGenerateSeeds(prng6_));
  cudaEventCreate(&event6_);
}

jams::BoseEinsteinCudaSRK4NoiseGenerator::~BoseEinsteinCudaSRK4NoiseGenerator() {
  if (prng5_ != nullptr) {
    curandDestroyGenerator(prng5_);
    prng5_ = nullptr;
  }

  if (prng6_ != nullptr) {
    curandDestroyGenerator(prng6_);
    prng6_ = nullptr;
  }

  if (event5_ != nullptr) {
    cudaEventDestroy(event5_);
    event5_ = nullptr;
  }

  if (event6_ != nullptr) {
    cudaEventDestroy(event6_);
    event6_ = nullptr;
  }
}

void jams::BoseEinsteinCudaSRK4NoiseGenerator::update() {
  if (!is_warmed_up_) {
    is_warmed_up_ = true;
    warmup(num_warm_up_steps_);
  }

  const int block_size = 128;
  const int grid_size = (num_channels_ + block_size - 1) / block_size;
  const double reduced_delta_tau = delta_tau_ * temperature_;

  CHECK_CURAND_STATUS(curandGenerateNormalDouble(prng5_, psi5_.device_data(), psi5_.size(), 0.0, 1.0));
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(prng6_, psi6_.device_data(), psi6_.size(), 0.0, 1.0));

  jams::stochastic_rk4_cuda_kernel<<<grid_size, block_size, 0, dev_stream5_.get()>>>(
      w5_.device_data(),
      v5_.device_data(),
      psi5_.device_data(),
      2.7189,
      5.0142,
      reduced_delta_tau,
      num_channels_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  jams::stochastic_rk4_cuda_kernel<<<grid_size, block_size, 0, dev_stream6_.get()>>>(
      w6_.device_data(),
      v6_.device_data(),
      psi6_.device_data(),
      1.2223,
      3.2974,
      reduced_delta_tau,
      num_channels_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  cudaEventRecord(event5_, dev_stream5_.get());
  cudaEventRecord(event6_, dev_stream6_.get());

  cudaStreamWaitEvent(cuda_stream_.get(), event5_, 0);
  cudaStreamWaitEvent(cuda_stream_.get(), event6_, 0);

  jams::stochastic_combination_cuda_kernel<<<grid_size, block_size, 0, cuda_stream_.get()>>>(
      noise_.device_data(),
      v5_.device_data(),
      v6_.device_data(),
      temperature_,
      num_channels_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void jams::BoseEinsteinCudaSRK4NoiseGenerator::warmup(const unsigned steps) {
  std::cout << "warming up noise generator " << steps << " steps @ "
            << this->temperature() << "K" << std::endl;

  for (auto i = 0U; i < steps; ++i) {
    update();
  }
}
