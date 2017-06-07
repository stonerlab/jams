// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_SEMI_IMPLICIT_LLG_KERNEL_H
#define JAMS_SOLVER_CUDA_SEMI_IMPLICIT_LLG_KERNEL_H

#include "jams/core/cuda_defs.h"

__constant__ double dev_dt;
__constant__ unsigned int dev_num_spins;

__global__ void cuda_semi_implicit_llg_kernelA
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
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned idx3 = 3*idx;

  if(idx < dev_num_spins) {
    double h[3];
    double s[3];
    double f[3];
    double fxs[3];
    double sxh[3];
    double norm,b2ff,fdots;

    for (unsigned n = 0; n < 3; ++n) {
      h[n] = (h_dev[idx3 + n] + noise_dev[idx3 + n]) * gyro_dev[idx];
    }

    for (unsigned n = 0; n < 3; ++n) {
      s[n] = s_dev[idx3 + n];
    }

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    f[0] = -0.5*dev_dt*( h[0] + alpha_dev[idx]*sxh[0] );
    f[1] = -0.5*dev_dt*( h[1] + alpha_dev[idx]*sxh[1] );
    f[2] = -0.5*dev_dt*( h[2] + alpha_dev[idx]*sxh[2] );

    b2ff = (f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    norm = 1.0/(1.0+b2ff);

    fdots = (f[0]*s[0]+f[1]*s[1]+f[2]*s[2]);

    fxs[0] = (f[1]*s[2] - f[2]*s[1]);
    fxs[1] = (f[2]*s[0] - f[0]*s[2]);
    fxs[2] = (f[0]*s[1] - f[1]*s[0]);

    s_dev[idx3] = 0.5*( s[0] + ( s[0]*(1.0-b2ff) + 2.0*(fxs[0]+f[0]*fdots) )*norm);
    s_dev[idx3+1] = 0.5*( s[1] + ( s[1]*(1.0-b2ff) + 2.0*(fxs[1]+f[1]*fdots) )*norm);
    s_dev[idx3+2] = 0.5*( s[2] + ( s[2]*(1.0-b2ff) + 2.0*(fxs[2]+f[2]*fdots) )*norm);
  }

}

__global__ void cuda_semi_implicit_llg_kernelB
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
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned idx3 = 3*idx;

  if(idx < dev_num_spins) {
    double h[3];
    double s[3];
    double so[3];
    double f[3];
    double fxs[3];
    double sxh[3];
    double norm,b2ff,fdots;

    for (unsigned n = 0; n < 3; ++n) {
      h[n] = (h_dev[idx3 + n] + noise_dev[idx3 + n]) * gyro_dev[idx];
    }

    for (unsigned n = 0; n < 3; ++n) {
      s[n] = s_dev[idx3 + n];
    }

    for (unsigned n = 0; n < 3; ++n) {
      so[n] = s_old_dev[idx3 + n];
    }

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    f[0] = -0.5*dev_dt*( h[0] + alpha_dev[idx]*sxh[0]);
    f[1] = -0.5*dev_dt*( h[1] + alpha_dev[idx]*sxh[1]);
    f[2] = -0.5*dev_dt*( h[2] + alpha_dev[idx]*sxh[2]);

    b2ff = (f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    norm = 1.0/(1.0+b2ff);

    fdots = (f[0]*so[0]+f[1]*so[1]+f[2]*so[2]);

    fxs[0] = (f[1]*so[2] - f[2]*so[1]);
    fxs[1] = (f[2]*so[0] - f[0]*so[2]);
    fxs[2] = (f[0]*so[1] - f[1]*so[0]);

    s_dev[idx3] = norm*( so[0]*(1.0-b2ff) + 2.0*(fxs[0]+f[0]*fdots) );
    s_dev[idx3+1] = norm*( so[1]*(1.0-b2ff) + 2.0*(fxs[1]+f[1]*fdots) );
    s_dev[idx3+2] = norm*( so[2]*(1.0-b2ff) + 2.0*(fxs[2]+f[2]*fdots) );
  }
}

#endif
