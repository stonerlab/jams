// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H
#define JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H

#include "core/cuda_defs.h"


__device__ inline void cuda_cross_product(const double b[3], const double c[3], double a[3]) {
  // a = b x c
  a[0] = b[1]*c[2] - b[2]*c[1];
  a[1] = b[2]*c[0] - b[0]*c[2];
  a[2] = b[0]*c[1] - b[1]*c[0];
}

__device__ inline void cuda_llg_rhs(const double alpha, const double s[3], const double sxh[3], double rhs[3]) {
  rhs[0] = sxh[0] + alpha * ( s[1]*sxh[2] - s[2]*sxh[1] );
  rhs[1] = sxh[1] + alpha * ( s[2]*sxh[0] - s[0]*sxh[2] );
  rhs[2] = sxh[2] + alpha * ( s[0]*sxh[1] - s[1]*sxh[0] );
}

__device__ inline double rnorm(const double s[3]) {
  return rsqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  // return 1.0/sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
}

__global__ void cuda_heun_llg_kernelA
(
  double * s_dev,
  double * ds_dt_dev,
  const double * s_old_dev,
  const double * h_dev,
  const double * noise_dev,
  const double * gyro_dev,
  const double * alpha_dev,
  const int num_spins,
  const double dt
)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if(idx < num_spins) {
    const int idx3 = 3*idx;
    double h[3], s[3], rhs[3], sxh[3];
    int n;

    for (n = 0; n < 3; ++n) {
      h[n] = ( h_dev[idx3 + n] + noise_dev[idx3 + n])*gyro_dev[idx];
    }

    for (n = 0; n < 3; ++n) {
      s[n] = s_dev[idx3 + n];
    }

    cuda_cross_product(s, h, sxh);

    cuda_llg_rhs(alpha_dev[idx], s, sxh, rhs);

    for (n = 0; n < 3; ++n) {
      ds_dt_dev[idx3 + n] = 0.5 * rhs[n];
    }

    for (n = 0; n < 3; ++n) {
      s[n] = s[n] + dt * rhs[n];
    }

    for (n = 0; n < 3; ++n) {
      s_dev[idx3 + n] = s[n]*rnorm(s);
    }
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
  const int num_spins,
  const double dt
)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if(idx < num_spins) {
    const int idx3 = 3*idx;
    double h[3], s[3], rhs[3], sxh[3];
    int n;

    for (n = 0; n < 3; ++n) {
      h[n] = ( h_dev[idx3 + n] + noise_dev[idx3 + n])*gyro_dev[idx];
    }

    for (n = 0; n < 3; ++n) {
      s[n] = s_dev[idx3 + n];
    }

    cuda_cross_product(s, h, sxh);

    cuda_llg_rhs(alpha_dev[idx], s, sxh, rhs);


    for (n = 0; n < 3; ++n) {
      ds_dt_dev[idx3 + n] = ds_dt_dev[idx3 + n] + 0.5 * rhs[n];
    }

    for (n = 0; n < 3; ++n) {
      s[n] = s_old_dev[idx3 + n] + dt * ds_dt_dev[idx3 + n];
    }

    for (n = 0; n < 3; ++n) {
      s_dev[idx3 + n] = s[n]*rnorm(s);
    }
  }
}

#endif
