// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <random>
#include <mutex>

#include "jams/helpers/utils.h"
#include "jams/cuda/cuda_array_kernels.h"

#include "cuda_langevin_bose.h"
#include "cuda_langevin_bose_kernel.h"

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
#include "jams/thermostats/cuda_langevin_bose.h"
#include "jams/thermostats/cuda_langevin_bose_kernel.h"

using namespace std;

CudaLangevinBoseThermostat::CudaLangevinBoseThermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins),
  debug_(false)
  {
   cout << "\n  initialising CUDA Langevin semi-quantum noise thermostat\n";

   config->lookupValue("thermostat.zero_point", do_zero_point_);

   double t_warmup = 1e-10; // 0.1 ns
   config->lookupValue("thermostat.warmup_time", t_warmup);

   omega_max_ = 25.0 * kTwoPi * kTHz;
   config->lookupValue("thermostat.w_max", omega_max_);

   double dt_thermostat = ::config->lookup("solver.t_step");
   delta_tau_ = (dt_thermostat * kBoltzmann) / kHBar;

   uint64_t dev_rng_seed = jams::random_generator()();


   cout << "    seed " << dev_rng_seed << "\n";
   cout << "    omega_max (THz) " << omega_max_ / (kTwoPi * kTHz) << "\n";
   cout << "    hbar*w/kB " << (kHBar * omega_max_) / (kBoltzmann) << "\n";
   cout << "    t_step " << dt_thermostat << "\n";
   cout << "    delta tau " << delta_tau_ << "\n";

   cout << "    initialising CUDA streams\n";

   if (cudaStreamCreate(&dev_stream_) != cudaSuccess) {
     jams_die("Failed to create CUDA stream in CudaLangevinBoseThermostat");
   }

   if (cudaStreamCreate(&dev_curand_stream_) != cudaSuccess) {
     jams_die("Failed to create CURAND stream in CudaLangevinBoseThermostat");
   }

   cout << "    initialising CURAND\n";

   CHECK_CURAND_STATUS(curandCreateGenerator(&thermostat_rng_, CURAND_RNG_PSEUDO_DEFAULT));
   CHECK_CURAND_STATUS(curandSetStream(thermostat_rng_, dev_curand_stream_));
   CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(thermostat_rng_, dev_rng_seed));
   CHECK_CURAND_STATUS(curandGenerateSeeds(thermostat_rng_));
   CHECK_CURAND_STATUS(curandGenerateNormalDouble(thermostat_rng_, eta0_.device_data(), eta0_.size(), 0.0, 1.0));
   CHECK_CURAND_STATUS(curandGenerateNormalDouble(thermostat_rng_, eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));
   CHECK_CURAND_STATUS(curandGenerateNormalDouble(thermostat_rng_, eta1b_.device_data(), eta1b_.size(), 0.0, 1.0));

   for (int i = 0; i < num_spins; ++i) {
     for (int j = 0; j < 3; ++j) {
        sigma_(i,j) = (kBoltzmann) *
            sqrt((2.0 * globals::alpha(i) * globals::mus(i)) / (kHBar * kGyromagneticRatio * kBohrMagneton));
     }
   }

   num_warm_up_steps_ = static_cast<unsigned>(t_warmup / dt_thermostat);


  zeta5_.resize(num_spins * 3, 0.0);
  zeta5p_.resize(num_spins * 3, 0.0);
  zeta6_.resize(num_spins * 3, 0.0);
  zeta6p_.resize(num_spins * 3, 0.0);
  eta1a_.resize(2 * num_spins * 3, 0.0);
  eta1b_.resize(2 * num_spins * 3, 0.0);

  if (do_zero_point_) {
    zeta0_.resize(4 * num_spins * 3, 0.0);
    eta0_.resize(4 * num_spins * 3, 0.0);
  }
}

void CudaLangevinBoseThermostat::update() {
  if (!is_warmed_up_) {
    is_warmed_up_ = true;
    warmup(num_warm_up_steps_);
  }

  int block_size = 96;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const double reduced_omega_max = (kHBar * omega_max_) / (kBoltzmann * this->temperature());
  const double reduced_delta_tau = delta_tau_ * this->temperature();
  const double temperature = this->temperature();

  swap(eta1a_, eta1b_);
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(thermostat_rng_, eta1a_.device_data(), eta1a_.size(), 0.0, 1.0));

  bose_coth_stochastic_process_cuda_kernel<<<grid_size, block_size, 0, dev_stream_ >>> (
    noise_.device_data(), zeta5_.device_data(), zeta5p_.device_data(), zeta6_.device_data(), zeta6p_.device_data(),
    eta1b_.device_data(), sigma_.device_data(), reduced_delta_tau, temperature, reduced_omega_max, globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  if (do_zero_point_) {
    CHECK_CURAND_STATUS(curandGenerateNormalDouble(thermostat_rng_, eta0_.device_data(), eta0_.size(), 0.0, 1.0));

    bose_zero_point_stochastic_process_cuda_kernel <<< grid_size, block_size, 0, dev_stream_ >>> (
        noise_.device_data(), zeta0_.device_data(), eta0_.device_data(), sigma_.device_data(), reduced_delta_tau,
        temperature, reduced_omega_max, globals::num_spins3);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
  }
}

CudaLangevinBoseThermostat::~CudaLangevinBoseThermostat() {
  if (thermostat_rng_ != nullptr) {
    CHECK_CURAND_STATUS(curandDestroyGenerator(thermostat_rng_))
  }

  if (dev_stream_ != nullptr) {
    CHECK_CUDA_STATUS(cudaStreamDestroy(dev_stream_));
  }

  if (dev_curand_stream_ != nullptr) {
    CHECK_CUDA_STATUS(cudaStreamDestroy(dev_curand_stream_));
  }
}

void CudaLangevinBoseThermostat::warmup(const unsigned steps) {
  cout << "warming up thermostat " << steps << " steps @ " << this->temperature() << "K" << std::endl;

  for (auto i = 0; i < steps; ++i) {
    update();
  }
}
