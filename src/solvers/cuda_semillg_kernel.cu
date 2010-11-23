#ifndef __CUDA_SEMILLG_KERNEL__
#define __CUDA_SEMILLG_KERNEL__

__global__ void cuda_semi_llg_kernelA
(
  double * s_dev,
  double * h_dev,
  float * w_dev,
  double * mus_dev,
  double * gyro_dev,
  double * alpha_dev,
  double h_app_x,
  double h_app_y,
  double h_app_z,
  int nspins,
  double dt
)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idx3 = 3*idx;

  if(idx < nspins) {
    double h[3];
    double s[3];
    double f[3];
    double fxs[3];
    double sxh[3];
    double norm,b2ff,fdots;

    double sigma = sqrt((2.0*(1.3806504E-23)*alpha_dev[idx])/(dt*mus_dev[idx]));

    h[0] = ( h_dev[idx3] + ( w_dev[idx]*sigma + h_app_x)*mus_dev[idx] )*gyro_dev[idx];
    h[1] = ( h_dev[idx3+1] + ( w_dev[nspins+idx+1]*sigma + h_app_y)*mus_dev[idx] )*gyro_dev[idx];
    h[2] = ( h_dev[idx3+2] + ( w_dev[2*nspins+idx+2]*sigma + h_app_z)*mus_dev[idx] )*gyro_dev[idx];

    s[0] = s_dev[idx3];
    s[1] = s_dev[idx3+1];
    s[2] = s_dev[idx3+2];

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    f[0] = -0.5*dt*( h[0] + alpha_dev[idx]*sxh[0] );
    f[1] = -0.5*dt*( h[1] + alpha_dev[idx]*sxh[1] );
    f[2] = -0.5*dt*( h[2] + alpha_dev[idx]*sxh[2] );

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

__global__ void cuda_semi_llg_kernelB
(
  double * s_dev,
  double * s_new_dev,
  double * h_dev,
  float * w_dev,
  double * mus_dev,
  double * gyro_dev,
  double * alpha_dev,
  double h_app_x,
  double h_app_y,
  double h_app_z,
  int nspins,
  double dt
)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  const int idx3 = 3*idx;

  if(idx < nspins) {
    double h[3];
    double s[3];
    double so[3];
    double f[3];
    double fxs[3];
    double sxh[3];
    double norm,b2ff,fdots;

    double sigma = sqrt((2.0*(1.3806504E-23)*alpha_dev[idx])/(dt*mus_dev[idx]));

    h[0] = ( h_dev[idx3] + ( w_dev[idx]*sigma + h_app_x)*mus_dev[idx] )*gyro_dev[idx];
    h[1] = ( h_dev[idx3+1] + ( w_dev[nspins+idx+1]*sigma + h_app_y)*mus_dev[idx] )*gyro_dev[idx];
    h[2] = ( h_dev[idx3+2] + ( w_dev[2*nspins+idx+2]*sigma + h_app_z)*mus_dev[idx] )*gyro_dev[idx];

    s[0] = s_dev[idx3];
    s[1] = s_dev[idx3+1];
    s[2] = s_dev[idx3+2];

    so[0] = s_new_dev[idx3];
    so[1] = s_new_dev[idx3+1];
    so[2] = s_new_dev[idx3+2];

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    f[0] = -0.5*dt*( h[0] + alpha_dev[idx]*sxh[0]);
    f[1] = -0.5*dt*( h[1] + alpha_dev[idx]*sxh[1]);
    f[2] = -0.5*dt*( h[2] + alpha_dev[idx]*sxh[2]);

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
