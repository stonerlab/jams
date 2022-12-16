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

using namespace std;

void CUDAHeunLLGSolver::initialize(const libconfig::Setting& settings)
{
  using namespace globals;

  // convert input in seconds to picoseconds for internal units
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;


  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  cout << "\ntimestep (ps) " << step_size_ << "\n";
  cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  std::string thermostat_name = jams::config_optional<string>(config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name));

  cout << "  thermostat " << thermostat_name.c_str() << "\n";

  cout << "done\n";

  // check if we need to use zero safe versions of the kernels (for |S| = 0)
  zero_safe_kernels_required_ = false;
  for (auto i = 0; i < globals::num_spins; ++i) {
    if (approximately_zero(Vec3{s(i,0), s(i,1), s(i,2)}, DBL_EPSILON)) {
      zero_safe_kernels_required_ = true;
      break;
    }
  }

  if (zero_safe_kernels_required_) {
    jams_warning("Some spins have zero length so zero safe kernels will be used.");
  }

  s_old_.resize(num_spins, 3);
  for (auto i = 0; i < num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_old_(i, j) = s(i, j);
    }
  }
}

void CUDAHeunLLGSolver::run()
{
  using namespace globals;
  double t0 = time_;

  const dim3 block_size = {84, 3, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 3, 1});

  cudaMemcpyAsync(s_old_.device_data(),           // void *               dst
             s.device_data(),               // const void *         src
             num_spins3*sizeof(double),   // size_t               count
             cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
             dev_stream_.get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();

  compute_fields();

  if (zero_safe_kernels_required_) {
    cuda_zero_safe_heun_llg_kernelA<<<grid_size, block_size>>>
      (s.device_data(), ds_dt.device_data(), s_old_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), step_size_, num_spins);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  } else {
    cuda_heun_llg_kernelA<<<grid_size, block_size>>>
      (s.device_data(), ds_dt.device_data(), s_old_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), step_size_, num_spins);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  double mid_time_step = step_size_;
  time_ = t0 + mid_time_step;

  compute_fields();

  if (zero_safe_kernels_required_) {
    cuda_zero_safe_heun_llg_kernelB<<<grid_size, block_size>>>
      (s.device_data(), ds_dt.device_data(), s_old_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), step_size_, num_spins);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  } else {
    cuda_heun_llg_kernelB<<<grid_size, block_size>>>
      (s.device_data(), ds_dt.device_data(), s_old_.device_data(),
       h.device_data(), thermostat_->device_data(),
       gyro.device_data(), mus.device_data(), alpha.device_data(), step_size_, num_spins);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }

  iteration_++;
  time_ = iteration_ * step_size_;
}

