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

jams::BoseEinsteinCudaSRK4Thermostat::BoseEinsteinCudaSRK4Thermostat(const double &temperature, const double &sigma, const int num_spins)
: Thermostat(temperature, sigma, num_spins) {
   std::cout << "\n  initialising CUDA Langevin semi-quantum noise thermostat\n";

   jams_warning("This thermostat is currently broken. Do not use for production work.");

   double warmup_time = 100.0e-12;
   globals::config->lookupValue("thermostat.warmup_time", warmup_time);
   warmup_time = warmup_time / 1e-12; // convert into ps

   const auto& solver_settings = globals::config->lookup("solver");
   auto solver_time_step = jams::config_required<double>(solver_settings, "t_step") / 1e-12;  // convert into ps

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
}

void jams::BoseEinsteinCudaSRK4Thermostat::update() {
  if (!is_warmed_up_) {
    is_warmed_up_ = true;
    warmup(num_warm_up_steps_);
  }

  // We can solve the two stochastic processes (5 and 6 in the Savin paper)
  // separately and so we put each one into a separate CUDA stream.
  // The final step where we combine them is done in the default stream which
  // will not execute until the other streams are complete.

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_stream5_.get()));
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), psi5_.device_data(), psi5_.size(), 0.0, 1.0));

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), dev_stream6_.get()));
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(jams::instance().curand_generator(), psi6_.device_data(), psi6_.size(), 0.0, 1.0));

  CHECK_CURAND_STATUS(curandSetStream(jams::instance().curand_generator(), nullptr));

  int block_size = 128;
  int grid_size = (globals::num_spins3 + block_size - 1) / block_size;

  const double reduced_delta_tau = delta_tau_ * temperature_;

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

  jams::stochastic_combination_cuda_kernel<<<grid_size, block_size>>>(
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
