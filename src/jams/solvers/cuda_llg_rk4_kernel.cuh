// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LLG_RK4_KERNEL_H
#define JAMS_SOLVER_CUDA_LLG_RK4_KERNEL_H

#include "jams/cuda/cuda_common.h"

#include "jams/cuda/cuda_device_rk4.cuh"


__global__ void cuda_llg_rk4_kernel
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
      h[n] = ((h_dev[3*idx + n] / mus_dev[idx]) + noise_dev[3*idx + n]);
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

    double sxsxh[3] = {
        (s[1] * sxh[2] - s[2] * sxh[1]),
        (s[2] * sxh[0] - s[0] * sxh[2]),
        (s[0] * sxh[1] - s[1] * sxh[0])
    };

    double rhs[3];
    for (auto n = 0; n < 3; ++n) {
      rhs[n] = -gyro_dev[idx] * (sxh[n] + alpha_dev[idx] * sxsxh[n]);
    }

    for (auto n = 0; n < 3; ++n) {
      k_dev[3 * idx + n] = rhs[n];
    }
  }
}

__global__ void cuda_llg_rk4_combination_kernel
    (
        double * s_dev,
        const double * s_old,
        const double * k1_dev,
        const double * k2_dev,
        const double * k3_dev,
        const double * k4_dev,
        const double dt,
        const unsigned dev_num_spins
    )
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_num_spins) {

    double s[3];
    for (auto n = 0; n < 3; ++n) {
      s[n] = s_old[3*idx + n] + dt * (k1_dev[3*idx + n] + 2*k2_dev[3*idx + n] + 2*k3_dev[3*idx + n] + k4_dev[3*idx + n]) / 6.0;
    }

    double recip_snorm = rsqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);

    for (auto n = 0; n < 3; ++n) {
      s_dev[3*idx + n] = s[n] * recip_snorm;
    }
  }
}


#endif
