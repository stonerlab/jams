// cuda_ll_lorentzian_rk4.cu                                                          -*-C++-*-
// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cuda_llg_rk4.h"

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

#include "cuda_llg_rk4_kernel.cuh"

using namespace std;

void CUDALLGRK4Solver::initialize(const libconfig::Setting& settings)
{
  using namespace globals;

  CudaSolver::initialize(settings);

  // convert input in seconds to picoseconds for internal units
  time_step_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;


  max_steps_ = static_cast<int>(t_max / time_step_);
  min_steps_ = static_cast<int>(t_min / time_step_);

  cout << "\ntimestep (ps) " << time_step_ << "\n";
  cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  cout << "timestep " << time_step_ << "\n";
  cout << "t_max " << t_max << " steps (" <<  max_steps_ << ")\n";
  cout << "t_min " << t_min << " steps (" << min_steps_ << ")\n";

  std::string thermostat_name = jams::config_optional<string>(config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  thermostat_ = Thermostat::create(thermostat_name);

  cout << "  thermostat " << thermostat_name.c_str() << "\n";

  cout << "done\n";

  s_old_.resize(num_spins, 3);
  for (auto i = 0; i < num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_old_(i, j) = s(i, j);
    }
  }

  k1_.resize(num_spins, 3);
  k2_.resize(num_spins, 3);
  k3_.resize(num_spins, 3);
  k4_.resize(num_spins, 3);

}

void CUDALLGRK4Solver::run()
{
  using namespace globals;

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cudaMemcpyAsync(s_old_.device_data(),           // void *               dst
                  s.device_data(),               // const void *         src
                  num_spins3*sizeof(double),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  dev_stream_.get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  thermostat_->set_temperature(physics_module_->temperature());
  thermostat_->update();

  compute_fields();

  // k1
  cuda_llg_rk4_kernel<<<grid_size, block_size>>>
      (s.device_data(), k1_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  double mid_time_step = 0.5 * time_step_;
  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k1_.device_data(), 1, s.device_data(), 1));

  compute_fields();

  // k2
  cuda_llg_rk4_kernel<<<grid_size, block_size>>>
      (s.device_data(), k2_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  mid_time_step = 0.5 * time_step_;
  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k2_.device_data(), 1, s.device_data(), 1));

  compute_fields();

  // k3
  cuda_llg_rk4_kernel<<<grid_size, block_size>>>
      (s.device_data(), k3_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  mid_time_step = time_step_;
  CHECK_CUBLAS_STATUS(cublasDcopy(jams::instance().cublas_handle(), globals::num_spins3, s_old_.device_data(), 1, s.device_data(), 1));
  CHECK_CUBLAS_STATUS(cublasDaxpy(jams::instance().cublas_handle(), globals::num_spins3, &mid_time_step, k3_.device_data(), 1, s.device_data(), 1));

  compute_fields();

  // k4
  cuda_llg_rk4_kernel<<<grid_size, block_size>>>
      (s.device_data(), k4_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_llg_rk4_combination_kernel<<<grid_size, block_size>>>
      (s.device_data(), s_old_.device_data(),
       k1_.device_data(), k2_.device_data(), k3_.device_data(), k4_.device_data(),
       time_step_, num_spins);

  iteration_++;
}

