// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cuda_llg_rk4.h"

#include <cuda_runtime.h>

#include "jams/core/globals.h"
#include "jams/core/thermostat.h"

#include "jams/cuda/cuda_common.h"

#include "jams/cuda/cuda_spin_ops.h"
#include "jams/solvers/llg_rhs.h"

namespace {

__global__ void cuda_llg_rk4_kernel(const double* spins, double* slopes, const jams::Real* fields,
                                    const jams::Real* noise, const jams::Real* gyro_eff,
                                    const jams::Real* inverse_moments, const jams::Real* damping,
                                    const unsigned int num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;
  const unsigned int offset = 3 * spin_index;

  const double spin[3] = {spins[offset], spins[offset + 1], spins[offset + 2]};
  double effective_field[3];
  for (int n = 0; n < 3; ++n) {
    const unsigned int index = offset + n;
    effective_field[n] = fields[index] * inverse_moments[spin_index] + noise[index];
  }

  jams::solvers::llg_rhs(spin, effective_field, gyro_eff[spin_index], damping[spin_index],
                         slopes + offset);
}

}  // namespace

void CUDALLGRK4Solver::function_kernel(jams::MultiArray<double, 2>& spins,
                                       jams::MultiArray<double, 2>& k, const jams::Real* noise,
                                       cudaStream_t stream) {
  const dim3 block_size = {64, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_llg_rk4_kernel<<<grid_size, block_size, 0, stream>>>(
      spins.device_data(), k.mutable_device_data(), globals::h.device_data(),
      noise, gyro_eff_.device_data(), globals::inv_mus.device_data(),
      globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CUDALLGRK4Solver::post_step(jams::MultiArray<double, 2>& spins, cudaStream_t stream) {
  jams::normalise_spins_cuda(spins, stream);
}
