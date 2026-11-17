#include "jams/solvers/cuda_gse_rk4.h"

#include <cuda_runtime.h>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/thermostat.h"
#include "jams/cuda/cuda_common.h"

namespace {

__global__ void cuda_gse_rk4_kernel(const double* spins, double* slopes, const jams::Real* fields,
                                    const jams::Real* noise, const jams::Real* gyro_eff,
                                    const jams::Real* inverse_moments, const jams::Real* damping,
                                    const unsigned num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (spin_index >= num_spins) return;
  const unsigned int offset = 3 * spin_index;

  double h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = fields[offset + n] * inverse_moments[spin_index];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = spins[offset + n];
  }

  double sxh[3] = {(s[1] * h[2] - s[2] * h[1]), (s[2] * h[0] - s[0] * h[2]),
                   (s[0] * h[1] - s[1] * h[0])};

  for (auto n = 0; n < 3; ++n) {
    slopes[offset + n] =
        -gyro_eff[spin_index] * (sxh[n] - damping[spin_index] * h[n]) +
        gyro_eff[spin_index] * noise[offset + n];
  }
}

}  // namespace

void CUDAGSERK4Solver::function_kernel(jams::MultiArray<double, 2>& spins,
                                       jams::MultiArray<double, 2>& k, const jams::Real* noise,
                                       cudaStream_t stream) {
  const dim3 block_size = {64, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cuda_gse_rk4_kernel<<<grid_size, block_size, 0, stream>>>(
      spins.device_data(), k.mutable_device_data(), globals::h.device_data(),
      noise, gyro_eff_.device_data(), globals::inv_mus.device_data(),
      globals::alpha.device_data(), globals::num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}
