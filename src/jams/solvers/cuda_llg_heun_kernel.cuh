// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H
#define JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/mixed_precision.h"

__global__ void cuda_heun_llg_kernelA
(
  double * s_dev,
  double * ds_dt_dev,
  const double * s_old_dev,
  const jams::Real * h_dev,
  const jams::Real * noise_dev,
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const double dev_dt,
  const unsigned dev_num_spins
)
{
  __shared__ double s[85 * 3];
  __shared__ jams::Real h[85 * 3];
  double rhs;

  const unsigned int tx3 = 3 * threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int p0 = tx3 + ty;
  const unsigned int p1 = tx3 + ((ty + 1) % 3);
  const unsigned int p2 = tx3 + ((ty + 2) % 3);

  const unsigned int gxy = IDX2D(idx, ty, 85, 3);

  if (idx < dev_num_spins && ty < 3) {
    h[p0] = ((h_dev[gxy] / mus_dev[idx]) + noise_dev[gxy]);
    s[p0] = s_dev[gxy];

    __syncthreads();

    rhs = - gyro_dev[idx] * ((s[p1] * h[p2] - s[p2] * h[p1])
        + alpha_dev[idx] * ( s[p1] * (s[p0] * h[p1] - s[p1] * h[p0])
                           - s[p2] * (s[p2] * h[p0] - s[p0] * h[p2]) ));

    ds_dt_dev[gxy] = 0.5 * rhs;

    s[p0] = s[p0] + dev_dt * rhs;

    __syncthreads();

    s_dev[gxy] = s[p0] * rnorm3d(s[tx3], s[tx3 + 1], s[tx3 + 2]);
  }
}

__global__ void cuda_heun_llg_kernelB
(
  double * s_dev,
  double * ds_dt_dev,
  const double * s_old_dev,
  const jams::Real * h_dev,
  const jams::Real * noise_dev,
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const double dev_dt,
  const unsigned dev_num_spins
)
{
  __shared__ double s[85 * 3];
  __shared__ jams::Real h[85 * 3];
  double rhs;

  const unsigned int tx3 = 3*threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int p0 = tx3 + ty;
  const unsigned int p1 = tx3 + ((ty + 1) % 3);
  const unsigned int p2 = tx3 + ((ty + 2) % 3);

  const unsigned int gxy = IDX2D(idx, ty, 85, 3);

  if (idx < dev_num_spins && ty < 3) {
    h[p0] = ((h_dev[gxy] / mus_dev[idx]) + noise_dev[gxy]);
    s[p0] = s_dev[gxy];

    __syncthreads();

    rhs = - gyro_dev[idx] * ((s[p1] * h[p2] - s[p2] * h[p1])
                             + alpha_dev[idx] * ( s[p1] * (s[p0] * h[p1] - s[p1] * h[p0])
                                                  - s[p2] * (s[p2] * h[p0] - s[p0] * h[p2]) ));


    ds_dt_dev[gxy] = ds_dt_dev[gxy] + 0.5 * rhs;

    s[p0] = s_old_dev[gxy] + dev_dt * ds_dt_dev[gxy];

    __syncthreads();
    s_dev[gxy] = s[p0] * rnorm3d(s[tx3], s[tx3 + 1], s[tx3 + 2]);
  }
}

#endif
