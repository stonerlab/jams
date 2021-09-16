// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LLG_RK4_KERNEL_H
#define JAMS_SOLVER_CUDA_LLG_RK4_KERNEL_H

#include "jams/cuda/cuda_common.h"

#include "jams/cuda/cuda_device_rk4.cuh"


__global__ void cuda_ll_lorentzian_rk4_kernel
(
  double * s_k_dev,
  double * w_k_dev,
  double * v_k_dev,
  const double * s_dev,
  const double * w_dev,
  const double * v_dev,
  const double * h_dev,
  const double * noise_dev,
  const double * gyro_dev,
  const double * mus_dev,
  const double * alpha_dev,
  const double lorentzian_omega,
  const double lorentzian_gamma,
  const double lorentzian_A,
  const unsigned dev_num_spins
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_num_spins) {
    double v[3];
    for (auto n = 0; n < 3; ++n) {
      v[n] = v_dev[3*idx + n];
    }

    double h[3];
    for (auto n = 0; n < 3; ++n) {
      h[n] = ((h_dev[3*idx + n] / mus_dev[idx]) + noise_dev[3*idx + n] + v[n]);
    }

    double s[3];
    for (auto n = 0; n < 3; ++n) {
      s[n] = s_dev[3*idx + n];
    }

    double w[3];
    for (auto n = 0; n < 3; ++n) {
      w[n] = w_dev[3*idx + n];
    }

    double sxh[3] = {
        (s[1] * h[2] - s[2] * h[1]),
        (s[2] * h[0] - s[0] * h[2]),
        (s[0] * h[1] - s[1] * h[0])
    };

    for (auto n = 0; n < 3; ++n) {
      s_k_dev[3 * idx + n] = -gyro_dev[idx] * sxh[n];
    }


    // TECHNICAL NOTE
    // --------------
    // The \vec{S} in equations in Anders arXiv:2009.00600v2 are dimensionful
    // in units of \hbar. So if the quantum spin is 1/2  then
    // |\vec{S}| = \hbar / 2. In JAMS we use unit vectors for spins. Therefore
    // the differential equations of the memory kernel need rewriting.
    // The simplest way is to realise that \gamma\vec{S} = \mu_s \vec{e} where
    // \vec{e} is the unit vector of a spin and \mu_s is the classical moment
    // \mu_s = g S \mu_B. Hence we can write:
    //
    // dW(t)/dt = -\omega_0^2 V(t) - \Gamma W(t) + A \mu_s e(t)
    //
    // Noting also that in Anders conventions the gyromagnetic ratio gamma is
    // negative i.e. their spins point in the opposite direction to their
    // magnetic moments (technically correct), but after substituting
    // \gamma\vec{S} = \mu_s \vec{e} we retain the positive sign on the final
    // term of dW/dt.
    //
    for (auto n = 0; n < 3; ++n) {
      w_k_dev[3 * idx + n] = -lorentzian_omega * lorentzian_omega * v[n] - lorentzian_gamma * w[n] + lorentzian_A * mus_dev[idx] * s[n];
    }

    for (auto n = 0; n < 3; ++n) {
      v_k_dev[3 * idx + n] = w[n];
    }

  }
}

__global__ void cuda_ll_lorentzian_rk4_combination_kernel
    (
        double * x_dev,
        const double * x_old,
        const double * k1_dev,
        const double * k2_dev,
        const double * k3_dev,
        const double * k4_dev,
        const double dt,
        const unsigned size
    )
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
      x_dev[idx] = x_old[idx] + dt * (k1_dev[idx] + 2*k2_dev[idx] + 2*k3_dev[idx] + k4_dev[idx]) / 6.0;
  }
}

__global__ void cuda_ll_lorentzian_rk4_combination_normalize_kernel
    (
        double * s_dev,
        const double * s_old,
        const double * k1_dev,
        const double * k2_dev,
        const double * k3_dev,
        const double * k4_dev,
        const double dt,
        const unsigned dev_num_spins
    )
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dev_num_spins) {

    double s[3];
    for (auto n = 0; n < 3; ++n) {
      s[n] = s_old[3*idx + n] + dt * (k1_dev[3*idx + n] + 2*k2_dev[3*idx + n] + 2*k3_dev[3*idx + n] + k4_dev[3*idx + n]) / 6.0;
    }

    double recip_snorm = rsqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);

    for (auto n = 0; n < 3; ++n) {
      s_dev[3*idx + n] = s[n] * recip_snorm;
    }
  }
}


#endif
