#ifndef __CUDA_HEUNLLG_KERNEL__
#define __CUDA_HEUNLLG_KERNEL__

__global__ void cuda_heun_llms_kernelA
(
  double * s_dev,
  float * sf_dev,
  double * s_new_dev,
  float * h_dev,
  float * w_dev,
  double * u_dev,
  double * u_new_dev,
  double * omega_corr_dev,
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
    double rhs[3];
    double sxh[3];
    double norm;

    float mus = mat_dev[idx*4];
    float gyro = mat_dev[idx*4+1];
    float alpha = mat_dev[idx*4+2];
    float sigma = mat_dev[idx*4+3];
	
    h[0] = double(( h_dev[idx3] + ( w_dev[idx3]*sigma + h_app_x)*mus )*gyro);
    h[1] = double(( h_dev[idx3+1] + ( w_dev[idx3+1]*sigma + h_app_y)*mus )*gyro);
    h[2] = double(( h_dev[idx3+2] + ( w_dev[idx3+2]*sigma + h_app_z)*mus )*gyro);

	#pragma unroll
	for(int i=0; i<3; ++i){
    	s[i] = s_dev[idx3+i];
	}
    
    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

	#pragma unroll
	for(int i=0; i<3; ++i){
    	s_new_dev[idx3+i] = s[i] + 0.5*dt*rhs[i];
	}

	#pragma unroll
	for(int i=0; i<3; ++i){
    	s[i] = s[i] + dt*rhs[i];
	}

    norm = 1.0/sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    
	#pragma unroll
	for(int i=0; i<3; ++i){
    	s_dev[idx3+i]   = s[i]*norm;
	}
	
	#pragma unroll
	for(int i=0; i<3; ++i){
    	sf_dev[idx3+i]   = float(s[i]*norm);
	}
	
	#pragma unroll
    for(int i=0; i<3; ++i){
		rhs[i] = u[idx3+i] - omega_corr_dev[i]*(w[idx3+i] + alpha*sxh[i]);
    }
	
	#pragma unroll
	for(int i=0; i<3; ++i){
		u_new_dev[idx3+i] = u_dev[idx3+i] + 0.5*dt*rhs[i];
		u_dev[idx3+i] = u_dev[idx3+i] + dt*rhs[i];
	}
	
  }
}

__global__ void cuda_heun_llms_kernelB
(
  double * s_dev,
  float * sf_dev,
  double * s_new_dev,
  float * h_dev,
  float * w_dev,
  double * u_dev,
  double * u_new_dev,
  double * omega_corr_dev,
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
    double rhs[3];
    double sxh[3];
    double norm;

    float mus = mat_dev[idx*4];
    float gyro = mat_dev[idx*4+1];
    float alpha = mat_dev[idx*4+2];
    float sigma = mat_dev[idx*4+3];

    h[0] = double(( h_dev[idx3] + ( w_dev[idx3]*sigma + h_app_x)*mus )*gyro);
    h[1] = double(( h_dev[idx3+1] + ( w_dev[idx3+1]*sigma + h_app_y)*mus )*gyro);
    h[2] = double(( h_dev[idx3+2] + ( w_dev[idx3+2]*sigma + h_app_z)*mus )*gyro);

	#pragma unroll
	for(int i=0; i<3; ++i){
    	s[i] = s_dev[idx3+i];
	}

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

	#pragma unroll
	for(int i=0; i<3; ++i){
    	s[i] = s_new_dev[idx3+i] + 0.5*dt*rhs[i];
	}

    norm = 1.0/sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);

	#pragma unroll
	for(int i=0; i<3; ++i){
    	s_dev[idx3+i]   = s[i]*norm;
	}

	#pragma unroll
	for(int i=0; i<3; ++i){
    	sf_dev[idx3+i]   = float(s[i]*norm);
	}
	
	#pragma unroll
    for(i=0; i<3; ++i) {
      u_dev[idx3+i] = u_new_dev[idx3+i] + 0.5*dt*(u_dev[idx3+i] - omega_corr_dev[i]*(w_dev[idx3+i] + alpha*sxh[i]));
    }

  }
}

#endif
