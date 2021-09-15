// cuda_ll_lorentzian_rk4.cu                                                          -*-C++-*-
// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cuda_ll_lorentzian_rk4.h"

#include <cuda.h>

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

#include "jams/solvers/cuda_ll_lorentzian_rk4_kernel.cuh"

using namespace std;

namespace {
  void cuda_rk4_internal_timestep(double* x, const double* x_old, const double* x_k, const int size, const double step) {
    // does an internal rk4 step by producing x = x_old + step * x_k
    CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), size, x_old, 1, x, 1));
    CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), size, &step, x_k, 1, x, 1));
  }
};

void CUDALLLorentzianRK4Solver::initialize(const libconfig::Setting& settings)
{
  using namespace globals;

  // convert input in seconds to picoseconds for internal units

  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

  lorentzian_gamma_ = kTwoPi * jams::config_required<double>(config->lookup("thermostat"), "lorentzian_gamma");
  lorentzian_omega_ = kTwoPi * jams::config_required<double>(config->lookup("thermostat"), "lorentzian_omega0");

  lorentzian_A_ =  (globals::alpha(0) * pow4(lorentzian_omega_)) / (lorentzian_gamma_);

  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  cout << "timestep (ps) " << step_size_ << "\n";
  cout << "t_max (ps) " << t_max << " steps (" <<  max_steps_ << ")\n";
  cout << "t_min (ps) " << t_min << " steps (" << min_steps_ << ")\n";

  std::string thermostat_name = jams::config_optional<string>(config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name));

  cout << "  thermostat " << thermostat_name.c_str() << "\n";

  cout << "done\n";

  s_old_.resize(num_spins, 3);
  for (auto i = 0; i < num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_old_(i, j) = s(i, j);
    }
  }

  s_k1_.resize(num_spins, 3);
  s_k2_.resize(num_spins, 3);
  s_k3_.resize(num_spins, 3);
  s_k4_.resize(num_spins, 3);

  zero(w_memory_process_.resize(num_spins, 3));
  zero(w_memory_process_old_.resize(num_spins, 3));
  zero(w_memory_process_k1_.resize(num_spins, 3));
  zero(w_memory_process_k2_.resize(num_spins, 3));
  zero(w_memory_process_k3_.resize(num_spins, 3));
  zero(w_memory_process_k4_.resize(num_spins, 3));

  zero(v_memory_process_.resize(num_spins, 3));
  zero(v_memory_process_old_.resize(num_spins, 3));
  zero(v_memory_process_k1_.resize(num_spins, 3));
  zero(v_memory_process_k2_.resize(num_spins, 3));
  zero(v_memory_process_k3_.resize(num_spins, 3));
  zero(v_memory_process_k4_.resize(num_spins, 3));

}

void CUDALLLorentzianRK4Solver::run()
{
  using namespace globals;

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cudaMemcpyAsync(s_old_.device_data(),           // void *               dst
                  s.device_data(),               // const void *         src
                  num_spins3*sizeof(double),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  dev_stream_.get());                   // device stream

  cudaMemcpyAsync(w_memory_process_old_.device_data(),           // void *               dst
                  w_memory_process_.device_data(),               // const void *         src
                  num_spins3*sizeof(double),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  dev_stream_.get());                   // device stream

  cudaMemcpyAsync(v_memory_process_old_.device_data(),           // void *               dst
                  v_memory_process_.device_data(),               // const void *         src
                  num_spins3*sizeof(double),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  dev_stream_.get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();

  cudaDeviceSynchronize();

  compute_fields();

  // k1
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size>>>(
      s_k1_.device_data(),
      w_memory_process_k1_.device_data(),
      v_memory_process_k1_.device_data(),
      s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      h.device_data(), thermostat_->device_data(),
      gyro.device_data(), mus.device_data(), alpha.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_rk4_internal_timestep(s.device_data(), s_old_.device_data(), s_k1_.device_data(), globals::num_spins3, 0.5 * step_size_);
  cuda_rk4_internal_timestep(w_memory_process_.device_data(), w_memory_process_old_.device_data(), w_memory_process_k1_.device_data(), globals::num_spins3, 0.5 * step_size_);
  cuda_rk4_internal_timestep(v_memory_process_.device_data(), v_memory_process_old_.device_data(), v_memory_process_k1_.device_data(), globals::num_spins3, 0.5 * step_size_);

  compute_fields();

  // k2
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size>>>(
      s_k2_.device_data(),
      w_memory_process_k2_.device_data(),
      v_memory_process_k2_.device_data(),
      s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      h.device_data(), thermostat_->device_data(),
      gyro.device_data(), mus.device_data(), alpha.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_rk4_internal_timestep(s.device_data(), s_old_.device_data(), s_k2_.device_data(), globals::num_spins3, 0.5 * step_size_);
  cuda_rk4_internal_timestep(w_memory_process_.device_data(), w_memory_process_old_.device_data(), w_memory_process_k2_.device_data(), globals::num_spins3, 0.5 * step_size_);
  cuda_rk4_internal_timestep(v_memory_process_.device_data(), v_memory_process_old_.device_data(), v_memory_process_k2_.device_data(), globals::num_spins3, 0.5 * step_size_);

  compute_fields();

  // k3
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size>>>(
      s_k3_.device_data(),
      w_memory_process_k3_.device_data(),
      v_memory_process_k3_.device_data(),
      s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      h.device_data(), thermostat_->device_data(),
      gyro.device_data(), mus.device_data(), alpha.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_rk4_internal_timestep(s.device_data(), s_old_.device_data(), s_k3_.device_data(), globals::num_spins3, step_size_);
  cuda_rk4_internal_timestep(w_memory_process_.device_data(), w_memory_process_old_.device_data(), w_memory_process_k3_.device_data(), globals::num_spins3, step_size_);
  cuda_rk4_internal_timestep(v_memory_process_.device_data(), v_memory_process_old_.device_data(), v_memory_process_k3_.device_data(), globals::num_spins3, step_size_);

  compute_fields();

  // k4
  cuda_ll_lorentzian_rk4_kernel<<<grid_size, block_size>>>(
      s_k4_.device_data(),
      w_memory_process_k4_.device_data(),
      v_memory_process_k4_.device_data(),
      s.device_data(),
      w_memory_process_.device_data(),
      v_memory_process_.device_data(),
      h.device_data(), thermostat_->device_data(),
      gyro.device_data(), mus.device_data(), alpha.device_data(),
      lorentzian_omega_, lorentzian_gamma_, lorentzian_A_,
      num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_ll_lorentzian_rk4_combination_normalize_kernel<<<grid_size, block_size>>>
      (s.device_data(), s_old_.device_data(),
       s_k1_.device_data(), s_k2_.device_data(), s_k3_.device_data(),
       s_k4_.device_data(),
       step_size_, num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  auto grid_size3 = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins3), 1, 1});

  cuda_ll_lorentzian_rk4_combination_kernel<<<grid_size3, block_size>>>(
      w_memory_process_.device_data(),
      w_memory_process_old_.device_data(),
      w_memory_process_k1_.device_data(),
      w_memory_process_k2_.device_data(),
      w_memory_process_k3_.device_data(),
      w_memory_process_k4_.device_data(),
      step_size_, num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_ll_lorentzian_rk4_combination_kernel<<<grid_size3, block_size>>>(
      v_memory_process_.device_data(),
      v_memory_process_old_.device_data(),
      v_memory_process_k1_.device_data(),
      v_memory_process_k2_.device_data(),
      v_memory_process_k3_.device_data(),
      v_memory_process_k4_.device_data(),
      step_size_, num_spins3);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  iteration_++;
}

