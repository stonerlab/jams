// cuda_gse_rk4.cu                                                     -*-C++-*-

#include <jams/solvers/cuda_gse_rk4.h>

#include <cuda_runtime.h>

#include <jams/common.h>
#include "jams/core/globals.h"
#include "jams/core/thermostat.h"
#include "jams/cuda/cuda_common.h"

#include "cuda_gse_rk4_kernel.cuh"

void CUDAGSERK4Solver::function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k) {
  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_gse_rk4_kernel<<<grid_size, block_size>>>
      (spins.device_data(), k.device_data(),
       globals::h.device_data(), thermostat_->device_data(),
       globals::gyro.device_data(), globals::mus.device_data(),
       globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

