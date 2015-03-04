// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H
#define JAMS_SOLVER_CUDA_HEUNLLG_KERNEL_H

#include "core/cuda_defs.h"

__global__ void cuda_heun_llg_kernelA
(
  double * s_dev,
  double * s_new_dev,
  CudaFastFloat * h_dev,
  const double * noise_dev,
  CudaFastFloat * mat_dev,
  CudaFastFloat h_app_x,
  CudaFastFloat h_app_y,
  CudaFastFloat h_app_z,
  int num_spins,
  double dt
)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idx3 = 3*idx;

  if(idx < num_spins) {
    double h[3];
    double s[3];
    double rhs[3];
    double sxh[3];
    double norm;

    CudaFastFloat mus = mat_dev[idx*4];
    CudaFastFloat gyro = mat_dev[idx*4+1];
    CudaFastFloat alpha = mat_dev[idx*4+2];
    CudaFastFloat sigma = mat_dev[idx*4+3];

    h[0] = ( h_dev[idx3] + ( noise_dev[idx3]*sigma + h_app_x)*mus )*gyro;
    h[1] = ( h_dev[idx3+1] + ( noise_dev[idx3+1]*sigma + h_app_y)*mus )*gyro;
    h[2] = ( h_dev[idx3+2] + ( noise_dev[idx3+2]*sigma + h_app_z)*mus )*gyro;

    s[0] = s_dev[idx3];
    s[1] = s_dev[idx3+1];
    s[2] = s_dev[idx3+2];

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    rhs[0] = sxh[0] + alpha * ( s[1]*sxh[2] - s[2]*sxh[1] );
    rhs[1] = sxh[1] + alpha * ( s[2]*sxh[0] - s[0]*sxh[2] );
    rhs[2] = sxh[2] + alpha * ( s[0]*sxh[1] - s[1]*sxh[0] );

    s_new_dev[idx3  ] = s[0] + 0.5*dt*rhs[0];
    s_new_dev[idx3+1] = s[1] + 0.5*dt*rhs[1];
    s_new_dev[idx3+2] = s[2] + 0.5*dt*rhs[2];

    s[0] = s[0] + dt*rhs[0];
    s[1] = s[1] + dt*rhs[1];
    s[2] = s[2] + dt*rhs[2];

    norm = 1.0/sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);

    s_dev[idx3]   = s[0]*norm;
    s_dev[idx3+1] = s[1]*norm;
    s_dev[idx3+2] = s[2]*norm;
  }
}

__global__ void cuda_heun_llg_kernelB
(
  double * s_dev,
  double * s_new_dev,
  CudaFastFloat * h_dev,
  const double * noise_dev,
  CudaFastFloat * mat_dev,
  CudaFastFloat h_app_x,
  CudaFastFloat h_app_y,
  CudaFastFloat h_app_z,
  int num_spins,
  double dt
)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idx3 = 3*idx;

  if(idx < num_spins) {
    double h[3];
    double s[3];
    double rhs[3];
    double sxh[3];
    double norm;

    CudaFastFloat mus = mat_dev[idx*4];
    CudaFastFloat gyro = mat_dev[idx*4+1];
    CudaFastFloat alpha = mat_dev[idx*4+2];
    CudaFastFloat sigma = mat_dev[idx*4+3];

    h[0] = ( h_dev[idx3] + ( noise_dev[idx3]*sigma + h_app_x)*mus )*gyro;
    h[1] = ( h_dev[idx3+1] + ( noise_dev[idx3+1]*sigma + h_app_y)*mus )*gyro;
    h[2] = ( h_dev[idx3+2] + ( noise_dev[idx3+2]*sigma + h_app_z)*mus )*gyro;

    s[0] = s_dev[idx3];
    s[1] = s_dev[idx3+1];
    s[2] = s_dev[idx3+2];

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    rhs[0] = sxh[0] + alpha * ( s[1]*sxh[2] - s[2]*sxh[1] );
    rhs[1] = sxh[1] + alpha * ( s[2]*sxh[0] - s[0]*sxh[2] );
    rhs[2] = sxh[2] + alpha * ( s[0]*sxh[1] - s[1]*sxh[0] );

    s[0] = s_new_dev[idx3  ] + 0.5*dt*rhs[0];
    s[1] = s_new_dev[idx3+1] + 0.5*dt*rhs[1];
    s[2] = s_new_dev[idx3+2] + 0.5*dt*rhs[2];

    norm = 1.0/sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);

    s_dev[idx3]   = s[0]*norm;
    s_dev[idx3+1] = s[1]*norm;
    s_dev[idx3+2] = s[2]*norm;
  }
}

#endif
