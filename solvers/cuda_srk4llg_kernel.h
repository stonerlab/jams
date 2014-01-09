// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_SRK4LLG_KERNEL_H
#define JAMS_SOLVER_CUDA_SRK4LLG_KERNEL_H

// IDEA: the double precision array s_dev could be allocated with a pitch,
// because we don't use if for field calculation anyway. Then the float array
// sf_dev can still be a 1d array for the field calculation.

// NOTE: const some of these arguments
__global__ void CUDAIntegrateLLG_SRK4
(
  double*      s_dev,
  double*      s_old_dev,
  double*      k_dev,
  float*       h_dev,
  float*       w_dev,
  float*       sf_dev,
  float*       mat_dev,
  float        h_app_x,
  float        h_app_y,
  float        h_app_z,
  double       q,
  double       dt,
  unsigned int nSpins
)
{
    const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int idx3 = 3*idx;


    if ( idx < nSpins ){
        double h[3];
        double s[3];
        double k[3];
        double sxh[3];

        float mus   = mat_dev[idx*4];
        float gyro  = mat_dev[idx*4+1];
        float alpha = mat_dev[idx*4+2];
        float sigma = mat_dev[idx*4+3];

        h[0] = double(( h_dev[idx3] + ( w_dev[idx3]*sigma + h_app_x)*mus )*gyro);
        h[1] = double(( h_dev[idx3+1] + ( w_dev[idx3+1]*sigma + h_app_y)*mus )*gyro);
        h[2] = double(( h_dev[idx3+2] + ( w_dev[idx3+2]*sigma + h_app_z)*mus )*gyro);

#pragma unroll
        for(int n=0; n<3; ++n){
            s[n] = s_old_dev[idx3+n];
        }

        // LLG
        sxh[0] = s[1]*h[2] - s[2]*h[1];
        sxh[1] = s[2]*h[0] - s[0]*h[2];
        sxh[2] = s[0]*h[1] - s[1]*h[0];

        k[0] = sxh[0] + alpha * ( s[1]*sxh[2] - s[2]*sxh[1] );
        k[1] = sxh[1] + alpha * ( s[2]*sxh[0] - s[0]*sxh[2] );
        k[2] = sxh[2] + alpha * ( s[0]*sxh[1] - s[1]*sxh[0] );

#pragma unroll
        for(int n=0; n<3; ++n){
            k_dev[idx3+n] = k[n];
        }

#pragma unroll
        for(int n=0; n<3; ++n){
            s[n] = s_dev[idx3+n] + q*dt*k[n];
        }

        double rnorm = rsqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);

#pragma unroll
        for(int n=0; n<3; ++n){
            sf_dev[idx3+n] = float(s[n]*rnorm);
        }

#pragma unroll
        for(int n=0; n<3; ++n){
            s_old_dev[idx3+n] = s[n]*rnorm;
        }
    }

}

__global__ void CUDAIntegrateEndPointLLG_SRK4
(
  double*      s_dev,
  double*      s_old_dev,
  double*      k0_dev,
  double*      k1_dev,
  double*      k2_dev,
  float*       h_dev,
  float*       w_dev,
  float*       sf_dev,
  float*       mat_dev,
  float        h_app_x,
  float        h_app_y,
  float        h_app_z,
  double       dt,
  unsigned int nSpins
)
{
    const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int idx3 = 3*idx;


    if ( idx < nSpins ){
        double h[3];
        double s[3];
        double k[3];
        double sxh[3];

        float mus   = mat_dev[idx*4];
        float gyro  = mat_dev[idx*4+1];
        float alpha = mat_dev[idx*4+2];
        float sigma = mat_dev[idx*4+3];

        h[0] = double(( h_dev[idx3] + ( w_dev[idx3]*sigma + h_app_x)*mus )*gyro);
        h[1] = double(( h_dev[idx3+1] + ( w_dev[idx3+1]*sigma + h_app_y)*mus )*gyro);
        h[2] = double(( h_dev[idx3+2] + ( w_dev[idx3+2]*sigma + h_app_z)*mus )*gyro);

#pragma unroll
        for(int n=0; n<3; ++n){
            s[n] = s_old_dev[idx3+n];
        }

        // LLG
        sxh[0] = s[1]*h[2] - s[2]*h[1];
        sxh[1] = s[2]*h[0] - s[0]*h[2];
        sxh[2] = s[0]*h[1] - s[1]*h[0];

        k[0] = sxh[0] + alpha * ( s[1]*sxh[2] - s[2]*sxh[1] );
        k[1] = sxh[1] + alpha * ( s[2]*sxh[0] - s[0]*sxh[2] );
        k[2] = sxh[2] + alpha * ( s[0]*sxh[1] - s[1]*sxh[0] );

#pragma unroll
        for(int n=0; n<3; ++n){
            s[n] = s_dev[idx3+n] + dt*( (1.0/6.0)*(k0_dev[idx3+n] + 2.0*k1_dev[idx3+n] + 2.0*k2_dev[idx3+n] + k[n]));
        }

        double rnorm = rsqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);

#pragma unroll
        for(int n=0; n<3; ++n){
            s[n] = s[n]*rnorm;
        }

#pragma unroll
        for(int n=0; n<3; ++n){
            s_dev[idx3+n] = s[n];
        }

#pragma unroll
        for(int n=0; n<3; ++n){
            s_old_dev[idx3+n] = s[n];
        }

#pragma unroll
        for(int n=0; n<3; ++n){
            sf_dev[idx3+n] = float(s[n]);
        }
    }

}

#endif
