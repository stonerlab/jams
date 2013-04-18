#ifndef __CUDA_HEUNLLBP_KERNEL__
#define __CUDA_HEUNLLBP_KERNEL__

__global__ void cuda_heun_llbp_kernelA
(
  double * s_dev,
  float * sf_dev,
  double * s_new_dev,
  float * h_dev,
  float * w_dev,
  double * u1_dev,
  double * u1_new_dev,
  double * u2_dev,
  double * u2_new_dev,
  float * tc_dev,
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
  int i;

  if(idx < nspins) {
    double h[3];
    double s[3];
    double rhs[3];
    double sxh[3];
    double norm;

    const float mus = mat_dev[idx*4];
    const float gyro = mat_dev[idx*4+1];
    const float alpha = mat_dev[idx*4+2];
    const float sigma = mat_dev[idx*4+3];
	
	const float tc1 = tc_dev[2*idx+0];
	const float tc2 = tc_dev[2*idx+1];
	
	const float ratio = (tc1*tc1)/(tc1*tc1 - tc2*tc2);
	
    h[0] = (( h_dev[idx3  ] + (u2_dev[idx3  ] - u1_dev[idx3  ] + h_app_x)*mus )*gyro);
    h[1] = (( h_dev[idx3+1] + (u2_dev[idx3+1] - u1_dev[idx3+1] + h_app_y)*mus )*gyro);
    h[2] = (( h_dev[idx3+2] + (u2_dev[idx3+2] - u1_dev[idx3+2] + h_app_z)*mus )*gyro);

	#pragma unroll
	for(i=0; i<3; ++i){
    	s[i] = s_dev[idx3+i];
	}
    
    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

	#pragma unroll
	for(i=0; i<3; ++i){
    	s_new_dev[idx3+i] = s[i] + 0.5*dt*sxh[i];
	}

	#pragma unroll
	for(i=0; i<3; ++i){
    	s[i] = s[i] + dt*sxh[i];
	}

    norm = 1.0/sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    
	#pragma unroll
	for(i=0; i<3; ++i){
    	s_dev[idx3+i]   = s[i]*norm;
	}
	
	#pragma unroll
	for(i=0; i<3; ++i){
    	sf_dev[idx3+i]   = float(s[i]*norm);
	}
	
	const double u1[3] = {u1_dev[idx3+0],u1_dev[idx3+1],u1_dev[idx3+2]};
    for(i=0; i<3; ++i) {
		rhs[i] = (sigma*w_dev[idx3+i] - u1[i] - ratio*alpha*sxh[i])/(tc1);
		u1_new_dev[idx3+i] = u1[i] + 0.5*dt*rhs[i];
      	u1_dev[idx3+i] = u1[i] + dt*rhs[i];
    }
	
	const double u2[3] = {u2_dev[idx3+0],u2_dev[idx3+1],u2_dev[idx3+2]};
    for(i=0; i<3; ++i) {
		rhs[i] = (sigma*w_dev[idx3+i] - u2[i] - ratio*alpha*sxh[i])/(tc2);
		u2_new_dev[idx3+i] = u2[i] + 0.5*dt*rhs[i];
      	u2_dev[idx3+i] = u2[i] + dt*rhs[i];
    }
  }
}

__global__ void cuda_heun_llbp_kernelB
(
    double * s_dev,
    float * sf_dev,
    double * s_new_dev,
    float * h_dev,
    float * w_dev,
    double * u1_dev,
    double * u1_new_dev,
    double * u2_dev,
    double * u2_new_dev,
    float * tc_dev,
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
  int i;

  if(idx < nspins) {
    double h[3];
    double s[3];
    double sxh[3];
    double norm;

    float mus = mat_dev[idx*4];
    float gyro = mat_dev[idx*4+1];
    float alpha = mat_dev[idx*4+2];
    float sigma = mat_dev[idx*4+3];
	const float tc1 = tc_dev[2*idx+0];
	const float tc2 = tc_dev[2*idx+1];
	
	const float ratio = (tc1*tc1)/(tc1*tc1 - tc2*tc2);

    h[0] = (( h_dev[idx3  ] + (u2_dev[idx3  ] - u1_dev[idx3  ] + h_app_x)*mus )*gyro);
    h[1] = (( h_dev[idx3+1] + (u2_dev[idx3+1] - u1_dev[idx3+1] + h_app_y)*mus )*gyro);
    h[2] = (( h_dev[idx3+2] + (u2_dev[idx3+2] - u1_dev[idx3+2] + h_app_z)*mus )*gyro);


	#pragma unroll
	for(i=0; i<3; ++i){
    	s[i] = s_dev[idx3+i];
	}

    sxh[0] = s[1]*h[2] - s[2]*h[1];
    sxh[1] = s[2]*h[0] - s[0]*h[2];
    sxh[2] = s[0]*h[1] - s[1]*h[0];

	#pragma unroll
	for(i=0; i<3; ++i){
    	s[i] = s_new_dev[idx3+i] + 0.5*dt*sxh[i];
	}

    norm = 1.0/sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);

	#pragma unroll
	for(i=0; i<3; ++i){
    	s_dev[idx3+i]   = s[i]*norm;
	}

	#pragma unroll
	for(i=0; i<3; ++i){
    	sf_dev[idx3+i]   = float(s[i]*norm);
	}
	
	#pragma unroll
    for(i=0; i<3; ++i) {
      u1_dev[idx3+i] = u1_new_dev[idx3+i]+0.5*dt*((sigma*w_dev[idx3+i] - u1_dev[idx3+i] - ratio*alpha*sxh[i])/(tc1));
    }
   
	#pragma unroll
    for(i=0; i<3; ++i) {
      u2_dev[idx3+i] = u2_new_dev[idx3+i]+0.5*dt*((sigma*w_dev[idx3+i] - u2_dev[idx3+i] - ratio*alpha*sxh[i])/(tc2));
    }

  }
}

#endif
