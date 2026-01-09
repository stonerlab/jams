// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H

#include <jams/helpers/mixed_precision.h>

__device__ inline void linear_ode(const jams::Real A[4], const jams::Real eta[4], const double z[4], double f[4]) {
  for (auto i = 0; i < 4; ++i) {
    f[i] = A[i] * (eta[i] - z[i]);
  }
}

__device__ inline void bose_ode(const jams::Real A[2], const jams::Real eta[2], const double z[2], double f[2]) {
  f[0] = z[1];
  f[1] = eta[0] - A[1] * A[1] * z[0] - A[0] * z[1];
}

template<unsigned N>
__device__ inline void
rk4_vectored(void ode(const jams::Real[N], const jams::Real[N], const double[N], double[N]), const double h, const jams::Real A[N],
             const jams::Real eta[N], double z[N]) {
  double k1[N], k2[N], k3[N], k4[N], f[N];
  double u[N];

  for (auto i = 0; i < N; ++i) {
    u[i] = z[i];
  }

  ode(A, eta, u, f);

  for (auto i = 0; i < N; ++i) {
    k1[i] = h * f[i];
  }

  // K2
  for (auto i = 0; i < N; ++i) {
    u[i] = z[i] + 0.5 * k1[i];
  }

  ode(A, eta, u, f);

  for (auto i = 0; i < N; ++i) {
    k2[i] = h * f[i];
  }

  // K3
  for (auto i = 0; i < N; ++i) {
    u[i] = z[i] + 0.5 * k2[i];
  }

  ode(A, eta, u, f);

  for (auto i = 0; i < N; ++i) {
    k3[i] = h * f[i];
  }

  // K4
  for (auto i = 0; i < N; ++i) {
    u[i] = z[i] + k3[i];
  }

  ode(A, eta, u, f);

  for (auto i = 0; i < N; ++i) {
    k4[i] = h * f[i];
  }

  for (auto i = 0; i < N; ++i) {
    z[i] = z[i] + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
  }
}

__global__ inline void cuda_thermostat_quantum_spde_zero_point_kernel
        (
                jams::Real *noise,
                double *zeta,
                const jams::Real *eta,
                const jams::Real *sigma,
                const jams::Real h,
                const jams::Real T,
                const jams::Real w_m,
                const int N
        ) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {

    const jams::Real c[4] =
            {jams::Real(1.043576) * w_m,
             jams::Real(0.177222) * w_m,
             jams::Real(0.050319) * w_m,
             jams::Real(0.010241) * w_m};


    const jams::Real lambda[4] =
            {jams::Real(1.763817) * w_m,
             jams::Real(0.394613) * w_m,
             jams::Real(0.103506) * w_m,
             jams::Real(0.015873) * w_m};

    double z[4];
    for (auto i = 0; i < 4; ++i) {
      z[i] = zeta[4*x + i];
    }

    jams::Real e[4];
    for (auto i = 0; i < 4; ++i) {
      e[i] = eta[4*x + i] * sqrtf(jams::Real(2.0) / (lambda[i] * h));
    }

    rk4_vectored<4>(linear_ode, h, lambda, e, z);

    for (auto i = 0; i < 4; ++i) {
      zeta[4 * x + i] = z[i];
    }

    double s0 = 0.0;
    for (auto i = 0; i < 4; ++i) {
      s0 += c[i] * (e[i] - z[i]);
    }

    noise[x] += T * sigma[x] * static_cast<jams::Real>(s0);
  }
}


__global__ inline void cuda_thermostat_quantum_spde_no_zero_kernel
        (
                jams::Real *noise,
                double *zeta5,
                double *zeta5p,
                double *zeta6,
                double *zeta6p,
                const jams::Real *eta,
                const jams::Real *sigma,
                const jams::Real h,
                const jams::Real T,
                const int N
        ) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {

    double s1 = 0.0;
    jams::Real e[2];
    double z[2];

    jams::Real gamma_omega[2] = {jams::Real(5.0142), jams::Real(2.7189)};

    e[0] = eta[x] * sqrtf(jams::Real(2.0) * gamma_omega[0] / h);
    e[1] = jams::Real(0.0);

    z[0] = zeta5[x];
    z[1] = zeta5p[x];

    rk4_vectored<2>(bose_ode, h, gamma_omega, e, z);

    zeta5[x] = z[0];
    zeta5p[x] = z[1];

    s1 += 1.8315 * z[0];

    //-------------------------------------------------

    gamma_omega[0] = jams::Real(3.2974);
    gamma_omega[1] = jams::Real(1.2223);

    e[0] = eta[N + x] * sqrtf(jams::Real(2.0) * gamma_omega[0] / h);
    e[1] = jams::Real(0.0);

    z[0] = zeta6[x];
    z[1] = zeta6p[x];

    rk4_vectored<2>(bose_ode, h, gamma_omega, e, z);

    zeta6[x] = z[0];
    zeta6p[x] = z[1];

    s1 += 0.3429 * z[0];

    noise[x] = T * sigma[x] * static_cast<jams::Real>(s1);
  }
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
