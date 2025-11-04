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

CudaThermostatQuantumSpde::CudaThermostatQuantumSpde(const double &temperature, const double &sigma, const double timestep, const int num_spins)
: Thermostat(temperature, sigma, timestep, num_spins),
  debug_(false)
  {
   std::cout << "\n  initialising quantum-spde-gpu thermostat\n";

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

   std::cout << "    initialising CUDA streams\n";

   if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
     throw jams::GeneralException("Failed to create CUDA stream in CudaLangevinBoseThermostat");
   }

   if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess) {
     throw jams::GeneralException("Failed to create CURAND stream in CudaLangevinBoseThermostat");
   }

   std::cout << "    initialising CURAND\n";

   CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_curand_stream_));
   if (do_zero_point_) {
     CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), eta0_.device_data(), eta0_.size(), 0.0, 1.0));
   }
   CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));
   CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), eta1b_.device_data(), eta1b_.size(), 0.0, 1.0));

    for (int i = 0; i < num_spins; ++i) {
      for (int j = 0; j < 3; ++j) {
        sigma_(i,j) = (kBoltzmannIU) * sqrt((2.0 * globals::alpha(i))
                                            / (kHBarIU * globals::gyro(i) * globals::mus(i)));
      }
    }

   num_warm_up_steps_ = static_cast<unsigned>(t_warmup / dt_thermostat);
}

void CudaThermostatQuantumSpde::update() {
  if (!is_warmed_up_) {
    warmup(num_warm_up_steps_);
    is_warmed_up_ = true;
  }

  if (this->temperature() == 0) {
    CHECK_CUDA_STATUS(cudaMemset(noise_.device_data(), 0, noise_.elements()*sizeof(double)));
    return;
  }

  int block_size = 96;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const double reduced_omega_max = (kHBarIU * omega_max_) / (kBoltzmannIU * this->temperature());
  const double reduced_delta_tau = delta_tau_ * this->temperature();
  const double temperature = this->temperature();

  swap(eta1a_, eta1b_);
  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_curand_stream_));
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));

  cuda_thermostat_quantum_spde_no_zero_kernel<<<grid_size, block_size, 0, dev_stream_ >>> (
    noise_.device_data(), zeta5_.device_data(), zeta5p_.device_data(), zeta6_.device_data(), zeta6p_.device_data(),
    eta1b_.device_data(), sigma_.device_data(), reduced_delta_tau, temperature, reduced_omega_max, globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  if (do_zero_point_) {
    CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_curand_stream_));
    CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), eta0_.device_data(), eta0_.size(), 0.0, 1.0));

    cuda_thermostat_quantum_spde_zero_point_kernel <<< grid_size, block_size, 0, dev_stream_ >>> (
        noise_.device_data(), zeta0_.device_data(), eta0_.device_data(), sigma_.device_data(), reduced_delta_tau,
        temperature, reduced_omega_max, globals::num_spins3);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
  }
}

CudaThermostatQuantumSpde::~CudaThermostatQuantumSpde() {
  if (dev_stream_ != nullptr) {
    cudaStreamDestroy(dev_stream_);
  }

  if (dev_curand_stream_ != nullptr) {
    cudaStreamDestroy(dev_curand_stream_);
  }
}

void CudaThermostatQuantumSpde::warmup(const unsigned steps) {
  std::cout << "warming up thermostat " << steps << " steps @ " << this->temperature() << "K" << std::endl;

  for (auto i = 0; i < steps; ++i) {
    update();
  }
}
