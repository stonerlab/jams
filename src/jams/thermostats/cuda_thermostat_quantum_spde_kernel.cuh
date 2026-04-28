// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H

#include <jams/helpers/mixed_precision.h>

__device__ inline double ou_linear_update(const double z, const jams::Real lambda, const jams::Real eta,
                                          const jams::Real h) {
  const double decay = exp(-static_cast<double>(lambda) * static_cast<double>(h));
  return static_cast<double>(eta) + (z - static_cast<double>(eta)) * decay;
}

__device__ inline void bose_exact_update(const jams::Real A[2], const jams::Real eta0,
                                         const jams::Real h, double z[2]) {
  const double gamma = static_cast<double>(A[0]);
  const double omega = static_cast<double>(A[1]);
  const double omega2 = omega * omega;
  const double alpha = 0.5 * gamma;
  const double decay = exp(-alpha * static_cast<double>(h));
  const double force_eq = static_cast<double>(eta0) / omega2;

  double y0 = z[0] - force_eq;
  double v0 = z[1];

  const double discriminant = omega2 - alpha * alpha;
  if (discriminant > 0.0) {
    const double beta = sqrt(discriminant);
    const double c = cos(beta * static_cast<double>(h));
    const double s = sin(beta * static_cast<double>(h));
    const double inv_beta = 1.0 / beta;

    const double y1 = decay * (y0 * c + (v0 + alpha * y0) * inv_beta * s);
    const double v1 = decay * (v0 * c - (alpha * v0 + omega2 * y0) * inv_beta * s);

    z[0] = y1 + force_eq;
    z[1] = v1;
    return;
  }

  if (discriminant < 0.0) {
    const double beta = sqrt(-discriminant);
    const double c = cosh(beta * static_cast<double>(h));
    const double s = sinh(beta * static_cast<double>(h));
    const double inv_beta = 1.0 / beta;

    const double y1 = decay * (y0 * c + (v0 + alpha * y0) * inv_beta * s);
    const double v1 = decay * (v0 * c - (alpha * v0 + omega2 * y0) * inv_beta * s);

    z[0] = y1 + force_eq;
    z[1] = v1;
    return;
  }

  const double y1 = decay * (y0 + (v0 + alpha * y0) * static_cast<double>(h));
  const double v1 = decay * (v0 - alpha * (v0 + alpha * y0) * static_cast<double>(h));

  z[0] = y1 + force_eq;
  z[1] = v1;
}

__global__ inline void cuda_thermostat_quantum_spde_zero_point_kernel
        (
                jams::Real *noise,
                double *zeta,
                const jams::Real *eta,
                const jams::Real *sigma,
                const jams::Real h_omega_max,
                const jams::Real zero_point_scale,
                const int N
        ) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {

    const jams::Real c[4] =
            {jams::Real(1.043576),
             jams::Real(0.177222),
             jams::Real(0.050319),
             jams::Real(0.010241)};


    const jams::Real lambda_h[4] =
            {jams::Real(1.763817) * h_omega_max,
             jams::Real(0.394613) * h_omega_max,
             jams::Real(0.103506) * h_omega_max,
             jams::Real(0.015873) * h_omega_max};

    double z[4];
    for (auto i = 0; i < 4; ++i) {
      z[i] = zeta[4*x + i];
    }

    jams::Real e[4];
    for (auto i = 0; i < 4; ++i) {
      e[i] = eta[4*x + i] * sqrtf(jams::Real(2.0) / lambda_h[i]);
    }

    for (auto i = 0; i < 4; ++i) {
      const double decay = exp(-static_cast<double>(lambda_h[i]));
      z[i] = static_cast<double>(e[i]) + (z[i] - static_cast<double>(e[i])) * decay;
    }

    for (auto i = 0; i < 4; ++i) {
      zeta[4 * x + i] = z[i];
    }

    double s0 = 0.0;
    for (auto i = 0; i < 4; ++i) {
      s0 += c[i] * (e[i] - z[i]);
    }

    noise[x] += sigma[x] * zero_point_scale * static_cast<jams::Real>(s0);
  }
}

__global__ void cuda_thermostat_quantum_spde_stationary_zero_point_kernel
        (
                double *zeta,
                const jams::Real *eta,
                const jams::Real h_omega_max,
                const int N
        ) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    const jams::Real lambda_h[4] =
            {jams::Real(1.763817) * h_omega_max,
             jams::Real(0.394613) * h_omega_max,
             jams::Real(0.103506) * h_omega_max,
             jams::Real(0.015873) * h_omega_max};

    for (auto i = 0; i < 4; ++i) {
      const double decay = exp(-static_cast<double>(lambda_h[i]));
      const double variance = 2.0 * (1.0 - decay) / (static_cast<double>(lambda_h[i]) * (1.0 + decay));
      zeta[4 * x + i] = static_cast<double>(eta[4 * x + i]) * sqrt(variance);
    }
  }
}

__global__ void cuda_thermostat_quantum_spde_stationary_no_zero_kernel
        (
                double *zeta5,
                double *zeta5p,
                double *zeta6,
                double *zeta6p,
                const jams::Real *eta5,
                const jams::Real *eta6,
                const double l5_00,
                const double l5_10,
                const double l5_11,
                const double l6_00,
                const double l6_10,
                const double l6_11,
                const int N
        ) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    const double g5_0 = static_cast<double>(eta5[x]);
    const double g5_1 = static_cast<double>(eta5[N + x]);
    zeta5[x] = l5_00 * g5_0;
    zeta5p[x] = l5_10 * g5_0 + l5_11 * g5_1;

    const double g6_0 = static_cast<double>(eta6[x]);
    const double g6_1 = static_cast<double>(eta6[N + x]);
    zeta6[x] = l6_00 * g6_0;
    zeta6p[x] = l6_10 * g6_0 + l6_11 * g6_1;
  }
}


__global__ void cuda_thermostat_quantum_spde_no_zero_kernel
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

    bose_exact_update(gamma_omega, e[0], h, z);

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

    bose_exact_update(gamma_omega, e[0], h, z);

    zeta6[x] = z[0];
    zeta6p[x] = z[1];

    s1 += 0.3429 * z[0];

    noise[x] = T * sigma[x] * static_cast<jams::Real>(s1);
  }
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
