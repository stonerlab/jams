// cuda_ll_lorentzian_rk4.cu                                                          -*-C++-*-
// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cuda_llg_rk4.h"

#include <jams/common.h>

#include <cuda_runtime.h>

#include "jams/core/globals.h"
#include "jams/core/thermostat.h"

#include "jams/cuda/cuda_common.h"
#include "jams/solvers/llg_spin_torque_terms.h"
#include "jams/solvers/solver_descriptor.h"

#include "cuda_llg_rk4_kernel.cuh"
#include <jams/cuda/cuda_spin_ops.h>

CUDALLGRK4Solver::CUDALLGRK4Solver(const libconfig::Setting &settings)
    : CudaRK4BaseSolver(settings) {
  extra_torque_ = jams::solvers::build_llg_spin_torque_field(
      settings, jams::solvers::describe_solver_setting(settings, *globals::config)).torque;
}


void CUDALLGRK4Solver::function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k) {
  compute_fields();

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_llg_rk4_kernel<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>
      (spins.device_data(), k.device_data(),
       globals::h.device_data(), thermostat_->device_data(), extra_torque_.device_data(),
       globals::gyro.device_data(), globals::mus.device_data(),
       globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}


void CUDALLGRK4Solver::post_step(jams::MultiArray<double, 2> &spins) {
  jams::normalise_spins_cuda(spins, jams::instance().cuda_master_stream().get());
}
