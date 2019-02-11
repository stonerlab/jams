// Copyright 2014 Joseph Barker. All rights reserved.

#include "cuda_llg_heun.h"

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

#include "cuda_llg_heun_kernel.cuh"

#include "jblib/containers/array.h"

using namespace std;

void CUDAHeunLLGSolver::initialize(const libconfig::Setting& settings)
{
  using namespace globals;

  CudaSolver::initialize(settings);

  time_step_ = jams::config_required<double>(settings, "t_step");
  double dt = time_step_ * kGyromagneticRatio;

  auto t_max = jams::config_required<double>(settings, "t_max");
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0);

  max_steps_ = static_cast<int>(t_max / time_step_);
  min_steps_ = static_cast<int>(t_min / time_step_);

  cout << "timestep " << time_step_ << "\n";
  cout << "t_max " << t_max << " steps (" <<  max_steps_ << ")\n";
  cout << "t_min " << t_min << " steps (" << min_steps_ << ")\n";

  cout << "  copy time_step to symbol\n";
  CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_dt, &dt, sizeof(double)));

  cout << "  copy num_spins to symbol\n";
  CHECK_CUDA_STATUS(cudaMemcpyToSymbol(dev_num_spins, &globals::num_spins, sizeof(unsigned int)));

  std::string thermostat_name = jams::config_optional<string>(config->lookup("solver"), "thermostat", jams::default_solver_gpu_thermostat);
  thermostat_ = Thermostat::create(thermostat_name);

  cout << "  thermostat " << thermostat_name.c_str() << "\n";

  cout << "done\n";

  // check if we need to use zero safe versions of the kernels (for |S| = 0)
  zero_safe_kernels_required_ = false;
  for (auto i = 0; i < globals::num_spins; ++i) {
    if (globals::s(i, 0) == 0.0 && globals::s(i, 1) == 0.0 && globals::s(i, 2) == 0.0) {
      zero_safe_kernels_required_ = true;
      break;
    }
  }

  if (zero_safe_kernels_required_) {
    jams_warning("Some spins have zero length so zero safe kernels will be used.");
  }
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;

  const dim3 block_size = {84, 3, 1};
  auto grid_size = cuda_grid_size(block_size, {globals::num_spins, 3, 1});

  cudaMemcpyAsync(dev_s_old_.data(),           // void *               dst
             dev_s_.data(),               // const void *         src
             num_spins3*sizeof(double),   // size_t               count
             cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
             dev_stream_.get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  thermostat_->set_temperature(physics_module_->temperature());
  thermostat_->update();

  compute_fields();

  if (zero_safe_kernels_required_) {
    cuda_zero_safe_heun_llg_kernelA<<<grid_size, block_size>>>
      (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
       dev_h_.data(), thermostat_->noise(),
       dev_gyro_.data(), dev_alpha_.data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  } else {
    cuda_heun_llg_kernelA<<<grid_size, block_size>>>
      (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
       dev_h_.data(), thermostat_->noise(),
       dev_gyro_.data(), dev_alpha_.data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  compute_fields();

  if (zero_safe_kernels_required_) {
    cuda_zero_safe_heun_llg_kernelB<<<grid_size, block_size>>>
      (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
       dev_h_.data(), thermostat_->noise(),
       dev_gyro_.data(), dev_alpha_.data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  } else {
    cuda_heun_llg_kernelB<<<grid_size, block_size>>>
      (dev_s_.data(), dev_ds_dt_.data(), dev_s_old_.data(),
       dev_h_.data(), thermostat_->noise(),
       dev_gyro_.data(), dev_alpha_.data());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  iteration_++;
}

