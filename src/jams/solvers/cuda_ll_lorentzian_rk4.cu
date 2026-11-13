// cuda_ll_lorentzian_rk4.cu                                                          -*-C++-*-
// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cuda_ll_lorentzian_rk4.h"

#include <jams/common.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <jams/interface/config.h>
#include <jams/helpers/defaults.h>

#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/thermostat.h"
#include "jams/core/physics.h"
#include "jams/helpers/error.h"
#include "jams/cuda/cuda_common.h"
#include "jams/core/lattice.h"

#include "jams/solvers/cuda_ll_lorentzian_rk4_kernel.cuh"
#include "jams/solvers/cuda_rk4_base_kernel.cuh"

void CUDALLLorentzianRK4Solver::initialize(const libconfig::Setting& settings)
{
  if (globals::lattice->num_materials() > 1) {
    throw std::runtime_error(
        "CUDALLLorentzianRK4Solver is only implemented for single material cells");
  }

  // convert input in seconds to picoseconds for internal units

  initialize_time(settings);

  lorentzian_gamma_ = kTwoPi * jams::config_required<double>(globals::config->lookup("thermostat"), "lorentzian_gamma");
  lorentzian_omega_ = kTwoPi * jams::config_required<double>(globals::config->lookup("thermostat"), "lorentzian_omega0");

  // In arXiv:2009.00600v2 Janet uses eta_G for the Gilbert damping, but this is
  // a **dimensionful** Gilbert damping (implied by Eq. (1) in the paper and
  // also explicitly mentioned). In JAMS alpha is the dimensionless Gilbert
  // damping. The difference is eta_G = alpha / (mu_s * gamma). It's important
  // that we convert here to get the scaling of the noice correct (i.e. it
  // converts Janet's equations into the JAMS convention).

  const int first_spin = globals::first_magnetic_spin();
  double eta_G = globals::alpha(first_spin) * globals::inv_mus(first_spin)
      / globals::gyro(first_spin);

  lorentzian_A_ =  (eta_G * pow4(lorentzian_omega_)) / (lorentzian_gamma_);

  std::string thermostat_name = jams::config_optional<std::string>(globals::config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name, this->time_step()));

  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  std::cout << "done\n";

  initialize_gyro_eff(settings, gyro_eff_);

  s_old_.resize(globals::num_spins, 3);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_old_(i, j) = globals::s(i, j);
    }
  }

  zero(s_k1_.resize(globals::num_spins, 3));
  zero(s_k2_.resize(globals::num_spins, 3));
  zero(s_k3_.resize(globals::num_spins, 3));
  zero(s_k4_.resize(globals::num_spins, 3));

  zero(w_memory_process_.resize(globals::num_spins, 3));
  zero(w_memory_process_old_.resize(globals::num_spins, 3));
  zero(w_memory_process_k1_.resize(globals::num_spins, 3));
  zero(w_memory_process_k2_.resize(globals::num_spins, 3));
  zero(w_memory_process_k3_.resize(globals::num_spins, 3));
  zero(w_memory_process_k4_.resize(globals::num_spins, 3));

  zero(v_memory_process_.resize(globals::num_spins, 3));
  zero(v_memory_process_old_.resize(globals::num_spins, 3));
  zero(v_memory_process_k1_.resize(globals::num_spins, 3));
  zero(v_memory_process_k2_.resize(globals::num_spins, 3));
  zero(v_memory_process_k3_.resize(globals::num_spins, 3));
  zero(v_memory_process_k4_.resize(globals::num_spins, 3));

}

void CUDALLLorentzianRK4Solver::run()
{
  const double t0 = time_;
  const double half_step = 0.5 * step_size_;
  const auto stream = jams::instance().cuda_master_stream().get();

  const dim3 block_size = {64, 1, 1};
  const auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});
  const auto grid_size3 = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins3), 1, 1});

  CHECK_CUDA_STATUS(cudaMemcpyAsync(s_old_.mutable_device_data(),
      globals::s.device_data(), globals::s.bytes(), cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA_STATUS(cudaMemcpyAsync(w_memory_process_old_.mutable_device_data(),
      w_memory_process_.device_data(), w_memory_process_.bytes(), cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA_STATUS(cudaMemcpyAsync(v_memory_process_old_.mutable_device_data(),
      v_memory_process_.device_data(), v_memory_process_.bytes(), cudaMemcpyDeviceToDevice, stream));
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();
  // Some thermostats also enqueue work on auxiliary streams.
  CHECK_CUDA_STATUS(cudaDeviceSynchronize());

  compute_fields();

  // k1
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size, 0, stream>>>(
      s_k1_.mutable_device_data(),
      w_memory_process_k1_.mutable_device_data(),
      v_memory_process_k1_.mutable_device_data(),
      globals::s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      globals::h.device_data(), thermostat_->device_data(),
      gyro_eff_.device_data(), globals::mus.device_data(), globals::inv_mus.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  time_ = t0 + half_step;

  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      half_step, s_old_.device_data(), s_k1_.device_data(), globals::s.mutable_device_data());
  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      half_step, w_memory_process_old_.device_data(), w_memory_process_k1_.device_data(), w_memory_process_.mutable_device_data());
  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      half_step, v_memory_process_old_.device_data(), v_memory_process_k1_.device_data(), v_memory_process_.mutable_device_data());

  record_spin_barrier_event();
  compute_fields();

  // k2
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size, 0, stream>>>(
      s_k2_.mutable_device_data(),
      w_memory_process_k2_.mutable_device_data(),
      v_memory_process_k2_.mutable_device_data(),
      globals::s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      globals::h.device_data(), thermostat_->device_data(),
      gyro_eff_.device_data(), globals::mus.device_data(), globals::inv_mus.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      half_step, s_old_.device_data(), s_k2_.device_data(), globals::s.mutable_device_data());
  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      half_step, w_memory_process_old_.device_data(), w_memory_process_k2_.device_data(), w_memory_process_.mutable_device_data());
  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      half_step, v_memory_process_old_.device_data(), v_memory_process_k2_.device_data(), v_memory_process_.mutable_device_data());

  record_spin_barrier_event();
  compute_fields();

  // k3
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size, 0, stream>>>(
      s_k3_.mutable_device_data(),
      w_memory_process_k3_.mutable_device_data(),
      v_memory_process_k3_.mutable_device_data(),
      globals::s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      globals::h.device_data(), thermostat_->device_data(),
      gyro_eff_.device_data(), globals::mus.device_data(), globals::inv_mus.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      step_size_, s_old_.device_data(), s_k3_.device_data(), globals::s.mutable_device_data());
  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      step_size_, w_memory_process_old_.device_data(), w_memory_process_k3_.device_data(), w_memory_process_.mutable_device_data());
  cuda_rk4_mid_step_kernel<<<grid_size3, block_size, 0, stream>>>(globals::num_spins3,
      step_size_, v_memory_process_old_.device_data(), v_memory_process_k3_.device_data(), v_memory_process_.mutable_device_data());

  time_ = t0 + step_size_;

  record_spin_barrier_event();
  compute_fields();

  // k4
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size, 0, stream>>>(
      s_k4_.mutable_device_data(),
      w_memory_process_k4_.mutable_device_data(),
      v_memory_process_k4_.mutable_device_data(),
      globals::s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      globals::h.device_data(), thermostat_->device_data(),
      gyro_eff_.device_data(), globals::mus.device_data(), globals::inv_mus.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);

  cuda_spin_normalisation_kernel<<<grid_size, block_size, 0, stream>>>
      (globals::s.mutable_device_data(), s_old_.device_data(),
       s_k1_.device_data(), s_k2_.device_data(), s_k3_.device_data(),
       s_k4_.device_data(),
       step_size_, globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_rk4_combination_kernel<<<grid_size3, block_size, 0, stream>>>(
      w_memory_process_.mutable_device_data(),
      w_memory_process_old_.device_data(),
      w_memory_process_k1_.device_data(),
      w_memory_process_k2_.device_data(),
      w_memory_process_k3_.device_data(),
      w_memory_process_k4_.device_data(),
      step_size_, globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_rk4_combination_kernel<<<grid_size3, block_size, 0, stream>>>(
      v_memory_process_.mutable_device_data(),
      v_memory_process_old_.device_data(),
      v_memory_process_k1_.device_data(),
      v_memory_process_k2_.device_data(),
      v_memory_process_k3_.device_data(),
      v_memory_process_k4_.device_data(),
      step_size_, globals::num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  record_spin_barrier_event();
  advance_time();
}
