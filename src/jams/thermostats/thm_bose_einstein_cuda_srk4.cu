// thm_bose_einstein_cuda_srk4.cu                                      -*-C++-*-

#include "jams/thermostats/thm_bose_einstein_cuda_srk4.h"
#include "jams/thermostats/thm_bose_einstein_cuda_srk4_kernel.cuh"

#include <jams/common.h>
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/interface/config.h"
#include "jams/cuda/cuda_common.h"

#include <curand.h>

#include <cmath>
#include <string>
#include <iostream>

jams::BoseEinsteinCudaSRK4Thermostat::BoseEinsteinCudaSRK4Thermostat(const double &temperature, const double &sigma, const double timestep, const int num_spins)
: Thermostat(temperature, sigma, timestep, num_spins) {
   std::cout << "\n  initialising CUDA Langevin semi-quantum noise thermostat\n";

   jams_warning("This thermostat is currently broken. Do not use for production work.");

   double warmup_time = 100.0e-12;
   globals::config->lookupValue("thermostat.warmup_time", warmup_time);
   warmup_time = warmup_time / 1e-12; // convert into ps

   const auto& solver_settings = globals::config->lookup("solver");
   auto solver_time_step = timestep;

   delta_tau_ = (solver_time_step * kBoltzmannIU) / kHBarIU;
   num_warm_up_steps_ = static_cast<unsigned>(warmup_time / solver_time_step);

   zero(w5_.resize(num_spins * 3));
   zero(v5_.resize(num_spins * 3));
   zero(w6_.resize(num_spins * 3));
   zero(v6_.resize(num_spins * 3));
   zero(psi5_.resize(2 * num_spins * 3));
   zero(psi6_.resize(2 * num_spins * 3));

   for (int i = 0; i < num_spins; ++i) {
     for (int j = 0; j < 3; ++j) {
       sigma_(i,j) = (kBoltzmannIU) * sqrt((2.0 * globals::alpha(i))
           / (kHBarIU * globals::gyro(i) * globals::mus(i)));
     }
   }

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


jams::BoseEinsteinCudaSRK4Thermostat::~BoseEinsteinCudaSRK4Thermostat() {
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


void jams::BoseEinsteinCudaSRK4Thermostat::update() {
  if (!is_warmed_up_) {
    is_warmed_up_ = true;
    warmup(num_warm_up_steps_);
  }

  int block_size = 128;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const double reduced_delta_tau = delta_tau_ * temperature_;

  // We can solve the two stochastic processes (5 and 6 in the Savin paper)
  // separately and so we put each one into a separate CUDA stream.
  // The final step where we combine them is done in the default stream which
  // will not execute until the other streams are complete.

  CHECK_CURAND_STATUS(curandGenerateNormalDouble(prng5_, psi5_.device_data(), psi5_.size(), 0.0, 1.0));
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(prng6_, psi6_.device_data(), psi6_.size(), 0.0, 1.0));

  jams::stochastic_rk4_cuda_kernel<<<grid_size, block_size, 0, dev_stream5_.get()>>>(
      w5_.device_data(),
      v5_.device_data(),
      psi5_.device_data(),
      2.7189,
      5.0142,
      reduced_delta_tau,
      globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  jams::stochastic_rk4_cuda_kernel<<<grid_size, block_size, 0, dev_stream6_.get()>>>(
      w6_.device_data(),
      v6_.device_data(),
      psi6_.device_data(),
      1.2223,
      3.2974,
      reduced_delta_tau,
      globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  cudaEventRecord(event5_, dev_stream5_.get());
  cudaEventRecord(event6_, dev_stream6_.get());

  cudaStreamWaitEvent(dev_stream5_.get(), event6_, 0);

  jams::stochastic_combination_cuda_kernel<<<grid_size, block_size, 0, dev_stream5_.get()>>>(
      noise_.device_data(),
      v5_.device_data(),
      v6_.device_data(),
      sigma_.device_data(),
      temperature_,
      globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
}

void jams::BoseEinsteinCudaSRK4Thermostat::warmup(const unsigned steps) {
  std::cout << "warming up thermostat " << steps << " steps @ ";
  std::cout << this->temperature() << "K" << std::endl;

  for (auto i = 0; i < steps; ++i) {
    update();
  }
}
