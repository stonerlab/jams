// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H
#define JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H

#include "core/cuda_defs.h"

__global__ void cuda_heun_llg_kernelA
(
  double * s_dev,
  double * ds_dt_dev,
  const double * s_old_dev,
  const double * h_dev,
  const double * noise_dev,
  const double * gyro_dev,
  const double * alpha_dev,
  const unsigned int num_spins,
  const double dt
)
{
  __shared__ double s[16 * 3];
  __shared__ double h[16 * 3];
  double rhs;

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int p0 = (3 * tx) + ty;
  const unsigned int p1 = (3 * tx) + ((ty + 1) % 3);
  const unsigned int p2 = (3 * tx) + ((ty + 2) % 3);

  const unsigned int gxy = 3 * idx + ty;

  if (idx < num_spins && ty < 3) {
    h[p0] = (h_dev[gxy] + noise_dev[gxy]) * gyro_dev[idx];
    s[p0] = s_dev[gxy];

    __syncthreads();

    rhs = (s[p1] * h[p2] - s[p2] * h[p1])
        + alpha_dev[idx] * ( s[p1] * (s[p0] * h[p1] - s[p1] * h[p0])
                           - s[p2] * (s[p2] * h[p0] - s[p0] * h[p2]) );

    ds_dt_dev[gxy] = 0.5 * rhs;

    s[p0] = s[p0] + dt * rhs;

    __syncthreads();

    s_dev[gxy] = s[p0] * rsqrt(s[3 * tx] * s[3 * tx] + s[3 * tx + 1] * s[3 * tx + 1] + s[3 * tx + 2] * s[3 * tx + 2] );
  }
}

__global__ void cuda_heun_llg_kernelB
(
  double * s_dev,
  double * ds_dt_dev,
  const double * s_old_dev,
  const double * h_dev,
  const double * noise_dev,
  const double * gyro_dev,
  const double * alpha_dev,
  const unsigned int num_spins,
  const double dt
)
{
  __shared__ double s[16 * 3];
  __shared__ double h[16 * 3];
  double rhs, ds_dt;

  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;

  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int p0 = (3 * tx) + ty;
  const unsigned int p1 = (3 * tx) + ((ty + 1) % 3);
  const unsigned int p2 = (3 * tx) + ((ty + 2) % 3);

  const unsigned int gxy = 3 * idx + ty;

  if (idx < num_spins && ty < 3) {
    h[p0] = (h_dev[gxy] + noise_dev[gxy]) * gyro_dev[idx];
    s[p0] = s_dev[gxy];

    __syncthreads();

    rhs = (s[p1] * h[p2] - s[p2] * h[p1])
        + alpha_dev[idx] * ( s[p1] * (s[p0] * h[p1] - s[p1] * h[p0])
                           - s[p2] * (s[p2] * h[p0] - s[p0] * h[p2]) );


    ds_dt = ds_dt_dev[gxy] + 0.5 * rhs;
    ds_dt_dev[gxy] = ds_dt;

    s[p0] = s_old_dev[gxy] + dt * ds_dt;

    __syncthreads();
    s_dev[gxy] = s[p0] * rsqrt(s[3 * tx] * s[3 * tx] + s[3 * tx + 1] * s[3 * tx + 1] + s[3 * tx + 2] * s[3 * tx + 2] );
  }
}

#endif
