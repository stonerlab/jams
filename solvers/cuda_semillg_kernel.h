#ifndef __CUDA_SEMILLG_KERNEL__
#define __CUDA_SEMILLG_KERNEL__

__global__ void cuda_semi_llg_kernelA
(
  double * s_dev,
  float * sf_dev,
  float * h_dev,
  float * w_dev,
  float * mat_dev,
  float h_app_x,
  float h_app_y,
  float h_app_z,
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

    float mus = mat_dev[idx*4];
    float gyro = mat_dev[idx*4+1];
    float alpha = mat_dev[idx*4+2];
    float sigma = mat_dev[idx*4+3];

    h[0] = double(( h_dev[idx3] + ( w_dev[idx3]*sigma + h_app_x)*mus )*gyro);
    h[1] = double(( h_dev[idx3+1] + ( w_dev[idx3+1]*sigma + h_app_y)*mus )*gyro);
    h[2] = double(( h_dev[idx3+2] + ( w_dev[idx3+2]*sigma + h_app_z)*mus )*gyro);

    s[0] = s_dev[idx3];
    s[1] = s_dev[idx3+1];
    s[2] = s_dev[idx3+2];

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    f[0] = -0.5*dt*( h[0] + alpha*sxh[0] );
    f[1] = -0.5*dt*( h[1] + alpha*sxh[1] );
    f[2] = -0.5*dt*( h[2] + alpha*sxh[2] );

    b2ff = (f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    norm = 1.0/(1.0+b2ff);

    fdots = (f[0]*s[0]+f[1]*s[1]+f[2]*s[2]);

    fxs[0] = (f[1]*s[2] - f[2]*s[1]);
    fxs[1] = (f[2]*s[0] - f[0]*s[2]);
    fxs[2] = (f[0]*s[1] - f[1]*s[0]);

    s[0] = 0.5*( s[0] + ( s[0]*(1.0-b2ff) + 2.0*(fxs[0]+f[0]*fdots) )*norm);
    s[1] = 0.5*( s[1] + ( s[1]*(1.0-b2ff) + 2.0*(fxs[1]+f[1]*fdots) )*norm);
    s[2] = 0.5*( s[2] + ( s[2]*(1.0-b2ff) + 2.0*(fxs[2]+f[2]*fdots) )*norm);

    s_dev[idx3]   = s[0];
    s_dev[idx3+1] = s[1];
    s_dev[idx3+2] = s[2];

    sf_dev[idx3] = float(s[0]);
    sf_dev[idx3+1] = float(s[1]);
    sf_dev[idx3+2] = float(s[2]);
  }
}

__global__ void cuda_semi_llg_kernelB
(
  double * s_dev,
  float * sf_dev,
  double * s_new_dev,
  float * h_dev,
  float * w_dev,
  float * mat_dev,
  float h_app_x,
  float h_app_y,
  float h_app_z,
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

    float mus = mat_dev[idx*4];
    float gyro = mat_dev[idx*4+1];
    float alpha = mat_dev[idx*4+2];
    float sigma = mat_dev[idx*4+3];

//    double sigma = sqrt((2.0*(1.3806504E-23)*alpha)/(dt*mus));

    h[0] = double(( h_dev[idx3] +  (w_dev[idx3]*sigma + h_app_x)*mus )*gyro);
    h[1] = double(( h_dev[idx3+1] +  (w_dev[idx3+1]*sigma + h_app_y)*mus )*gyro);
    h[2] = double(( h_dev[idx3+2] +  (w_dev[idx3+2]*sigma + h_app_z)*mus )*gyro);

    s[0] = s_dev[idx3];
    s[1] = s_dev[idx3+1];
    s[2] = s_dev[idx3+2];

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

    f[0] = -0.5*dt*( h[0] + alpha*sxh[0]);
    f[1] = -0.5*dt*( h[1] + alpha*sxh[1]);
    f[2] = -0.5*dt*( h[2] + alpha*sxh[2]);

    b2ff = (f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    norm = 1.0/(1.0+b2ff);

    so[0] = s_new_dev[idx3];
    so[1] = s_new_dev[idx3+1];
    so[2] = s_new_dev[idx3+2];

    fdots = (f[0]*so[0]+f[1]*so[1]+f[2]*so[2]);

    fxs[0] = (f[1]*so[2] - f[2]*so[1]);
    fxs[1] = (f[2]*so[0] - f[0]*so[2]);
    fxs[2] = (f[0]*so[1] - f[1]*so[0]);

    s[0] = norm*( so[0]*(1.0-b2ff) + 2.0*(fxs[0]+f[0]*fdots) );
    s[1] = norm*( so[1]*(1.0-b2ff) + 2.0*(fxs[1]+f[1]*fdots) );
    s[2] = norm*( so[2]*(1.0-b2ff) + 2.0*(fxs[2]+f[2]*fdots) );

    s_dev[idx3]   = s[0];
    s_dev[idx3+1] = s[1];
    s_dev[idx3+2] = s[2];

    sf_dev[idx3] = float(s[0]);
    sf_dev[idx3+1] = float(s[1]);
    sf_dev[idx3+2] = float(s[2]);
  }
}

#endif
