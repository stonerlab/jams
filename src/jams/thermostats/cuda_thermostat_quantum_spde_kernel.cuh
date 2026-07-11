// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H

#include <jams/helpers/mixed_precision.h>
#include <jams/thermostats/cuda_quantum_spde_noise.h>

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

__device__ inline double zero_point_update_component(
    double *__restrict__ zeta,
    const jams::Real *__restrict__ eta,
    const jams::QuantumSpdeZeroPointUpdateCoefficients coeffs,
    const int component,
    const int x) {
  const double z_old = zeta[x];
  const double e = static_cast<double>(eta[x] * coeffs.eta_scale[component]);
  const double z_new = coeffs.decay[component] * z_old
      + (1.0 - coeffs.decay[component]) * e;
  zeta[x] = z_new;
  return static_cast<double>(coeffs.weight[component]) * (e - z_new);
}

__global__ void cuda_thermostat_quantum_spde_zero_point_fast_kernel
        (
                jams::Real *__restrict__ noise,
                double *__restrict__ zeta0,
                double *__restrict__ zeta1,
                double *__restrict__ zeta2,
                double *__restrict__ zeta3,
                const jams::Real *__restrict__ eta0,
                const jams::Real *__restrict__ eta1,
                const jams::Real *__restrict__ eta2,
                const jams::Real *__restrict__ eta3,
                const jams::Real *__restrict__ sigma,
                const jams::QuantumSpdeZeroPointUpdateCoefficients coeffs,
                const int N
        ) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    const double s0 =
        zero_point_update_component(zeta0, eta0, coeffs, 0, x)
        + zero_point_update_component(zeta1, eta1, coeffs, 1, x)
        + zero_point_update_component(zeta2, eta2, coeffs, 2, x)
        + zero_point_update_component(zeta3, eta3, coeffs, 3, x);

    noise[x] += sigma[x] * coeffs.zero_point_scale * static_cast<jams::Real>(s0);
  }
}

__device__ inline jams::Real zero_point_lambda_factor(const int component) {
  switch (component) {
    case 0:
      return jams::Real(1.763817);
    case 1:
      return jams::Real(0.394613);
    case 2:
      return jams::Real(0.103506);
    default:
      return jams::Real(0.015873);
  }
}

__device__ inline void stationary_zero_point_component(
    double *__restrict__ zeta,
    const jams::Real *__restrict__ eta,
    const jams::Real h_omega_max,
    const int component,
    const int x) {
  const double lambda_h = static_cast<double>(
      zero_point_lambda_factor(component) * h_omega_max);
  const double decay = exp(-lambda_h);
  const double variance = 2.0 * (1.0 - decay) / (lambda_h * (1.0 + decay));
  zeta[x] = static_cast<double>(eta[x]) * sqrt(variance);
}

__global__ void cuda_thermostat_quantum_spde_stationary_zero_point_kernel
        (
                double *__restrict__ zeta0,
                double *__restrict__ zeta1,
                double *__restrict__ zeta2,
                double *__restrict__ zeta3,
                const jams::Real *__restrict__ eta0,
                const jams::Real *__restrict__ eta1,
                const jams::Real *__restrict__ eta2,
                const jams::Real *__restrict__ eta3,
                const jams::Real h_omega_max,
                const int N
        ) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    stationary_zero_point_component(zeta0, eta0, h_omega_max, 0, x);
    stationary_zero_point_component(zeta1, eta1, h_omega_max, 1, x);
    stationary_zero_point_component(zeta2, eta2, h_omega_max, 2, x);
    stationary_zero_point_component(zeta3, eta3, h_omega_max, 3, x);
  }
}

__global__ void cuda_thermostat_quantum_spde_stationary_no_zero_kernel
        (
                double *zeta5,
                double *zeta5p,
                double *zeta6,
                double *zeta6p,
                const jams::Real *eta5_0,
                const jams::Real *eta5_1,
                const jams::Real *eta6_0,
                const jams::Real *eta6_1,
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
    const double g5_0 = static_cast<double>(eta5_0[x]);
    const double g5_1 = static_cast<double>(eta5_1[x]);
    zeta5[x] = l5_00 * g5_0;
    zeta5p[x] = l5_10 * g5_0 + l5_11 * g5_1;

    const double g6_0 = static_cast<double>(eta6_0[x]);
    const double g6_1 = static_cast<double>(eta6_1[x]);
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
                const jams::Real *eta5,
                const jams::Real *eta6,
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

    e[0] = eta5[x] * sqrtf(jams::Real(2.0) * gamma_omega[0] / h);
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

    e[0] = eta6[x] * sqrtf(jams::Real(2.0) * gamma_omega[0] / h);
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

__global__ void cuda_thermostat_quantum_spde_no_zero_fast_kernel
        (
                jams::Real *__restrict__ noise,
                double *__restrict__ zeta5,
                double *__restrict__ zeta5p,
                double *__restrict__ zeta6,
                double *__restrict__ zeta6p,
                const jams::Real *__restrict__ eta5,
                const jams::Real *__restrict__ eta6,
                const jams::Real *__restrict__ sigma,
                const jams::QuantumSpdeBoseUpdateCoefficients factor5,
                const jams::QuantumSpdeBoseUpdateCoefficients factor6,
                const jams::Real T,
                const int N
	        ) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    const double force5 = static_cast<double>(eta5[x] * factor5.eta_scale) * factor5.inv_omega2;
    const double z5 = zeta5[x];
    const double z5p = zeta5p[x];
    const double z5_new = factor5.m00 * z5 + factor5.m01 * z5p + factor5.force0 * force5;
    const double z5p_new = factor5.m10 * z5 + factor5.m11 * z5p + factor5.force1 * force5;
    zeta5[x] = z5_new;
    zeta5p[x] = z5p_new;

    const double force6 = static_cast<double>(eta6[x] * factor6.eta_scale) * factor6.inv_omega2;
    const double z6 = zeta6[x];
    const double z6p = zeta6p[x];
    const double z6_new = factor6.m00 * z6 + factor6.m01 * z6p + factor6.force0 * force6;
    const double z6p_new = factor6.m10 * z6 + factor6.m11 * z6p + factor6.force1 * force6;
    zeta6[x] = z6_new;
    zeta6p[x] = z6p_new;

    const double s1 = 1.8315 * z5_new + 0.3429 * z6_new;
    noise[x] = T * sigma[x] * static_cast<jams::Real>(s1);
  }
}

__global__ void cuda_thermostat_quantum_spde_stationary_no_zero_profile_kernel
        (
                double *__restrict__ zeta5,
                double *__restrict__ zeta5p,
                double *__restrict__ zeta6,
                double *__restrict__ zeta6p,
                const jams::Real *__restrict__ eta5_0,
                const jams::Real *__restrict__ eta5_1,
                const jams::Real *__restrict__ eta6_0,
                const jams::Real *__restrict__ eta6_1,
                const jams::QuantumSpdeBoseCholesky *__restrict__ factor5,
                const jams::QuantumSpdeBoseCholesky *__restrict__ factor6,
                const int N
        ) {

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    const auto f5 = factor5[x];
    const auto f6 = factor6[x];

    if (f5.l00 == 0.0 && f6.l00 == 0.0) {
      zeta5[x] = 0.0;
      zeta5p[x] = 0.0;
      zeta6[x] = 0.0;
      zeta6p[x] = 0.0;
      return;
    }

    const double g5_0 = static_cast<double>(eta5_0[x]);
    const double g5_1 = static_cast<double>(eta5_1[x]);
    zeta5[x] = f5.l00 * g5_0;
    zeta5p[x] = f5.l10 * g5_0 + f5.l11 * g5_1;

    const double g6_0 = static_cast<double>(eta6_0[x]);
    const double g6_1 = static_cast<double>(eta6_1[x]);
    zeta6[x] = f6.l00 * g6_0;
    zeta6p[x] = f6.l10 * g6_0 + f6.l11 * g6_1;
  }
}

__global__ void cuda_thermostat_quantum_spde_no_zero_profile_fast_kernel
        (
                jams::Real *__restrict__ noise,
                double *__restrict__ zeta5,
                double *__restrict__ zeta5p,
                double *__restrict__ zeta6,
                double *__restrict__ zeta6p,
                const jams::Real *__restrict__ eta5,
                const jams::Real *__restrict__ eta6,
                const jams::Real *__restrict__ sigma,
                const jams::QuantumSpdeBoseUpdateCoefficients *__restrict__ factor5,
                const jams::QuantumSpdeBoseUpdateCoefficients *__restrict__ factor6,
                const jams::Real *__restrict__ temperature,
                const int N
        ) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    const jams::Real T = temperature[x];
    if (T <= jams::Real{0.0}) {
      noise[x] = jams::Real{0.0};
      return;
    }

    const auto f5 = factor5[x];
    const auto f6 = factor6[x];

    const double force5 = static_cast<double>(eta5[x] * f5.eta_scale) * f5.inv_omega2;
    const double z5 = zeta5[x];
    const double z5p = zeta5p[x];
    const double z5_new = f5.m00 * z5 + f5.m01 * z5p + f5.force0 * force5;
    const double z5p_new = f5.m10 * z5 + f5.m11 * z5p + f5.force1 * force5;
    zeta5[x] = z5_new;
    zeta5p[x] = z5p_new;

    const double force6 = static_cast<double>(eta6[x] * f6.eta_scale) * f6.inv_omega2;
    const double z6 = zeta6[x];
    const double z6p = zeta6p[x];
    const double z6_new = f6.m00 * z6 + f6.m01 * z6p + f6.force0 * force6;
    const double z6p_new = f6.m10 * z6 + f6.m11 * z6p + f6.force1 * force6;
    zeta6[x] = z6_new;
    zeta6p[x] = z6p_new;

    const double s1 = 1.8315 * z5_new + 0.3429 * z6_new;
    noise[x] = T * sigma[x] * static_cast<jams::Real>(s1);
  }
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_BOSE_KERNEL_H
