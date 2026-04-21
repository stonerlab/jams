// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LLG_RK4_KERNEL_H
#define JAMS_SOLVER_CUDA_LLG_RK4_KERNEL_H

#include "jams/cuda/cuda_common.h"

#include "jams/cuda/cuda_device_rk4.cuh"
#include "jams/helpers/mixed_precision.h"


__global__ void cuda_llg_rk4_kernel
(
  const double * s_dev,
  double * k_dev,
  const jams::Real * h_dev,
  const jams::Real * noise_dev,
  const double * torque_dev,
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
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

    double tau[3];
    for (auto n = 0; n < 3; ++n) {
      tau[n] = torque_dev[3*idx + n];
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

    const double s_dot_tau = s[0] * tau[0] + s[1] * tau[1] + s[2] * tau[2];
    const double s_dot_s = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
    double sxsxtau[3] = {
        s_dot_tau * s[0] - s_dot_s * tau[0],
        s_dot_tau * s[1] - s_dot_s * tau[1],
        s_dot_tau * s[2] - s_dot_s * tau[2]
    };

    double rhs[3];
    for (auto n = 0; n < 3; ++n) {
      rhs[n] = -gyro_dev[idx] * (sxh[n] + alpha_dev[idx] * sxsxh[n] + sxsxtau[n] / mus_dev[idx]);
    }

    for (auto n = 0; n < 3; ++n) {
      k_dev[3 * idx + n] = rhs[n];
    }
  }
}



#endif
