// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H
#define JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H

#include "jams/cuda/cuda_defs.h"

__constant__ double dev_dt;
__constant__ unsigned int dev_num_spins;

__global__ void cuda_heun_llg_kernelA
(
  double * __restrict__ s_dev,
  double * __restrict__ ds_dt_dev,
  const double * __restrict__ s_old_dev,
  const double * __restrict__ h_dev,
  const double * __restrict__ noise_dev,
  const double * __restrict__ gyro_dev,
  const double * __restrict__ alpha_dev
)
{
  __shared__ double s[85 * 3];
  __shared__ double h[85 * 3];
  double rhs;

  const unsigned int tx3 = 3 * threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int p0 = tx3 + ty;
  const unsigned int p1 = tx3 + ((ty + 1) % 3);
  const unsigned int p2 = tx3 + ((ty + 2) % 3);

  const unsigned int gxy = 3 * idx + ty;

  if (idx < dev_num_spins && ty < 3) {
    h[p0] = (h_dev[gxy] + noise_dev[gxy]) * gyro_dev[idx];
    s[p0] = s_dev[gxy];

    __syncthreads();

    rhs = (s[p1] * h[p2] - s[p2] * h[p1])
        + alpha_dev[idx] * ( s[p1] * (s[p0] * h[p1] - s[p1] * h[p0])
                           - s[p2] * (s[p2] * h[p0] - s[p0] * h[p2]) );

    ds_dt_dev[gxy] = 0.5 * rhs;

    s[p0] = s[p0] + dev_dt * rhs;

    __syncthreads();

    s_dev[gxy] = s[p0] * rsqrt(s[tx3] * s[tx3] + s[tx3 + 1] * s[tx3 + 1] + s[tx3 + 2] * s[tx3 + 2] );
  }
}

__global__ void cuda_heun_llg_kernelB
(
  double * __restrict__ s_dev,
  double * __restrict__ ds_dt_dev,
  const double * __restrict__ s_old_dev,
  const double * __restrict__ h_dev,
  const double * __restrict__ noise_dev,
  const double * __restrict__ gyro_dev,
  const double * __restrict__ alpha_dev
)
{
  __shared__ double s[85 * 3];
  __shared__ double h[85 * 3];
  double rhs;

  const unsigned int tx3 = 3*threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int p0 = tx3 + ty;
  const unsigned int p1 = tx3 + ((ty + 1) % 3);
  const unsigned int p2 = tx3 + ((ty + 2) % 3);

  const unsigned int gxy = 3 * idx + ty;

  if (idx < dev_num_spins && ty < 3) {
    h[p0] = (h_dev[gxy] + noise_dev[gxy]) * gyro_dev[idx];
    s[p0] = s_dev[gxy];

    __syncthreads();

    rhs = (s[p1] * h[p2] - s[p2] * h[p1])
        + alpha_dev[idx] * ( s[p1] * (s[p0] * h[p1] - s[p1] * h[p0])
                           - s[p2] * (s[p2] * h[p0] - s[p0] * h[p2]) );


    ds_dt_dev[gxy] = ds_dt_dev[gxy] + 0.5 * rhs;

    s[p0] = s_old_dev[gxy] + dev_dt * ds_dt_dev[gxy];

    __syncthreads();
    s_dev[gxy] = s[p0] * rsqrt(s[tx3] * s[tx3] + s[tx3 + 1] * s[tx3 + 1] + s[tx3 + 2] * s[tx3 + 2] );
  }
}

#endif
