#ifndef JAMS_SOLVER_CUDA_RK4_LLG_SOT_KERNEL_H
#define JAMS_SOLVER_CUDA_RK4_LLG_SOT_KERNEL_H

#include <jams/cuda/cuda_device_vector_ops.h>

__global__ void cuda_rk4_llg_sot_kernel
    (
        const double * s_dev,
        double * k_dev,
        const double * h_dev,
        const double * polarisation_dev,
        const double * sot_dev,
        const double * noise_dev,
        const double * gyro_dev,
        const double * mus_dev,
        const double * alpha_dev,
        const unsigned dev_num_spins
    )
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_num_spins) {

    double h[3];
    for (auto n = 0; n < 3; ++n) {
      h[n] = ((h_dev[3*idx + n] / mus_dev[idx]) + noise_dev[3*idx + n]);
    }

    double s[3];
    for (auto n = 0; n < 3; ++n) {
      s[n] = s_dev[3*idx + n];
    }

    double j[3];
    for (auto n = 0; n < 3; ++n) {
      j[n] = polarisation_dev[3*idx + n];
    }

    double sxh[3] = {
        (s[1] * h[2] - s[2] * h[1]),
        (s[2] * h[0] - s[0] * h[2]),
        (s[0] * h[1] - s[1] * h[0])
    };

    double sxsxh[3] = {
        (s[1] * sxh[2] - s[2] * sxh[1]),
        (s[2] * sxh[0] - s[0] * sxh[2]),
        (s[0] * sxh[1] - s[1] * sxh[0])
    };

    // using vector triple product
    // s x s x j = (s . j) s - (s . s) j
    // and accepting that s may not be a unit vector

    double s_dot_j = dot(s, j);
    double s_dot_s = dot(s, s);

    double sxsxj[3] = {
        s_dot_j * s[0] - s_dot_s * j[0],
        s_dot_j * s[1] - s_dot_s * j[1],
        s_dot_j * s[2] - s_dot_s * j[2]
    };

    for (auto n = 0; n < 3; ++n) {
      k_dev[3 * idx + n] = -gyro_dev[idx] * (sxh[n] + alpha_dev[idx] * sxsxh[n] + (sot_dev[idx] / mus_dev[idx]) * sxsxj[n]);
    }
  }
}



#endif
