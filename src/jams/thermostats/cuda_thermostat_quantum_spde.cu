// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <random>
#include <mutex>

#include <jams/common.h>
#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_array_kernels.h"

#include "jams/thermostats/cuda_thermostat_quantum_spde.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde_kernel.cuh"

#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/cuda/cuda_array_kernels.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/helpers/random.h"
#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_common.h"
#include "jams/monitors/magnetisation.h"
#include <jams/helpers/exception.h>

CudaThermostatQuantumSpde::CudaThermostatQuantumSpde(const jams::Real &temperature, const jams::Real &sigma, const jams::Real timestep, const int num_spins)
: Thermostat(temperature, sigma, timestep, num_spins)
  {
   std::cout << "\n  initialising quantum-spde-gpu thermostat\n";

  cudaEventCreateWithFlags(&curand_done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_stream_ = CudaStream(CudaStream::Priority::LOW);

  zeta5_.resize(num_spins * 3).zero();
  zeta5p_.resize(num_spins * 3).zero();
  zeta6_.resize(num_spins * 3).zero();
  zeta6p_.resize(num_spins * 3).zero();
  eta1a_.resize(2 * num_spins * 3).zero();
  eta1b_.resize(2 * num_spins * 3).zero();

   globals::config->lookupValue("thermostat.zero_point", do_zero_point_);
   if (do_zero_point_) {
     zeta0_.resize(4 * num_spins * 3).zero();
     eta0_.resize(4 * num_spins * 3).zero();
   }

   double t_warmup = 1e-10 / 1e-12; // 0.1 ns
   globals::config->lookupValue("thermostat.warmup_time", t_warmup);

   omega_max_ = 25.0 * kTwoPi;
   globals::config->lookupValue("thermostat.w_max", omega_max_);

   double dt_thermostat = timestep;
   delta_tau_ = (dt_thermostat * kBoltzmannIU) / kHBarIU;

   std::cout << "    omega_max (THz) " << omega_max_ / (kTwoPi) << "\n";
   std::cout << "    hbar*w/kB " << (kHBarIU * omega_max_) / (kBoltzmannIU) << "\n";
   std::cout << "    t_step " << dt_thermostat << "\n";
   std::cout << "    delta tau " << delta_tau_ << "\n";


   if (do_zero_point_) {
     CHECK_CURAND_STATUS(curandGenerateNormal(jams::instance().curand_generator(), eta0_.device_data(), eta0_.size(), 0.0, 1.0));
   }
   CHECK_CURAND_STATUS(curandGenerateNormal(jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));
   CHECK_CURAND_STATUS(curandGenerateNormal(jams::instance().curand_generator(), eta1b_.device_data(), eta1b_.size(), 0.0, 1.0));

  cudaEventRecord(curand_done_, curand_stream_.get());

    for (int i = 0; i < num_spins; ++i) {
      for (int j = 0; j < 3; ++j) {
        sigma_(i,j) = static_cast<jams::Real>((kBoltzmannIU) * sqrt((2.0 * globals::alpha(i))
                                            / (kHBarIU * globals::gyro(i) * globals::mus(i))));
      }
    }

   auto num_warm_up_steps = static_cast<unsigned>(t_warmup / dt_thermostat);

  std::cout << "warming up thermostat " << num_warm_up_steps << " steps @ " << this->temperature() << "K" << std::endl;
  for (auto i = 0; i < num_warm_up_steps; ++i) {
    CudaThermostatQuantumSpde::update();
  }
}

void CudaThermostatQuantumSpde::update() {
  if (this->temperature() == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise_.device_data(), 0, noise_.bytes(), cuda_stream_.get()));
    return;
  }

  int block_size = 128;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const double reduced_omega_max = (kHBarIU * omega_max_) / (kBoltzmannIU * this->temperature());
  const double reduced_delta_tau = delta_tau_ * this->temperature();
  const jams::Real temperature = this->temperature();

  // We swap pointers to arrays of random numbers and then generate new numbers AFTER the kernel invocation. This
  // allows the expensive generation to be hidden by multiplexing with other streams all the way until we next get back
  // to the thermostat update.
  swap(eta1a_, eta1b_);

  cudaStreamWaitEvent(cuda_stream_.get(), curand_done_, 0);
  cuda_thermostat_quantum_spde_no_zero_kernel<<<grid_size, block_size, 0, cuda_stream_.get() >>> (
    noise_.device_data(), zeta5_.device_data(), zeta5p_.device_data(), zeta6_.device_data(), zeta6p_.device_data(),
    eta1b_.device_data(), sigma_.device_data(), reduced_delta_tau, temperature, globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));
#endif

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS


  if (do_zero_point_) {
    cuda_thermostat_quantum_spde_zero_point_kernel <<< grid_size, block_size, 0, cuda_stream_.get() >>> (
        noise_.device_data(), zeta0_.device_data(), eta0_.device_data(), sigma_.device_data(), reduced_delta_tau,
        temperature, reduced_omega_max, globals::num_spins3);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;

    CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), curand_stream_.get()));
#ifdef DO_MIXED_PRECISION
    CHECK_CURAND_STATUS(curandGenerateNormal(jams::instance().curand_generator(), eta0_.device_data(), eta0_.size(), 0.0, 1.0));
#else
    CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), eta0_.device_data(), eta0_.size(), 0.0, 1.0));
#endif

    cudaEventRecord(curand_done_, curand_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
}