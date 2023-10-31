#ifndef JAMS_SOLVER_CUDA_GSE_RK4_KERNEL_H
#define JAMS_SOLVER_CUDA_GSE_RK4_KERNEL_H

#include "jams/cuda/cuda_common.h"

#include "jams/cuda/cuda_device_rk4.cuh"


__global__ void cuda_gse_rk4_kernel
(
  const double * s_dev,
  double * k_dev,
  const double * h_dev,
  const double * noise_dev,
  const double * gyro_dev,
  const double * mus_dev,
  const double * alpha_dev,
  const unsigned dev_num_spins
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_num_spins) {

    double h[3];
    for (auto n = 0; n < 3; ++n) {
      h[n] = h_dev[3*idx + n] / mus_dev[idx];
    }

    double s[3];
    for (auto n = 0; n < 3; ++n) {
      s[n] = s_dev[3*idx + n];
    }

    double sxh[3] = {
        (s[1] * h[2] - s[2] * h[1]),
        (s[2] * h[0] - s[0] * h[2]),
        (s[0] * h[1] - s[1] * h[0])
    };

    for (auto n = 0; n < 3; ++n) {
      k_dev[3 * idx + n] = -gyro_dev[idx] * (sxh[n] - alpha_dev[idx] * h[n]) + gyro_dev[idx] * noise_dev[3*idx + n];
    }
  }
}


#endif
