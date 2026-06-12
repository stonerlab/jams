// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVERS_LLG_RKMK_FUNCTIONS_H
#define JAMS_SOLVERS_LLG_RKMK_FUNCTIONS_H

#include <cmath>

#ifndef JAMS_HOST_DEVICE
#if defined(__CUDACC__)
#define JAMS_HOST_DEVICE __host__ __device__
#else
#define JAMS_HOST_DEVICE
#endif
#endif

#ifndef JAMS_FORCEINLINE
#if defined(__CUDACC__)
#define JAMS_FORCEINLINE __forceinline__
#else
#define JAMS_FORCEINLINE inline
#endif
#endif

namespace jams::solvers::rkmk {

JAMS_HOST_DEVICE JAMS_FORCEINLINE double dot3(const double a[3],
                                              const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

JAMS_HOST_DEVICE JAMS_FORCEINLINE double norm_squared3(const double v[3]) {
  return dot3(v, v);
}

JAMS_HOST_DEVICE JAMS_FORCEINLINE void cross_product3(const double a[3],
                                                      const double b[3],
                                                      double out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

JAMS_HOST_DEVICE JAMS_FORCEINLINE void sincos_scalar(const double x,
                                                     double* s,
                                                     double* c) {
#if defined(__CUDA_ARCH__)
  sincos(x, s, c);
#else
  *s = std::sin(x);
  *c = std::cos(x);
#endif
}

template <typename FieldReal, typename GyroReal, typename AlphaReal>
JAMS_HOST_DEVICE JAMS_FORCEINLINE void omega_llg(const double s[3],
                                                 const FieldReal h[3],
                                                 const GyroReal gyro,
                                                 const AlphaReal alpha,
                                                 double out[3]) {
  out[0] = static_cast<double>(gyro) *
      (static_cast<double>(h[0]) +
       static_cast<double>(alpha) *
           (s[1] * static_cast<double>(h[2]) -
            s[2] * static_cast<double>(h[1])));
  out[1] = static_cast<double>(gyro) *
      (static_cast<double>(h[1]) +
       static_cast<double>(alpha) *
           (s[2] * static_cast<double>(h[0]) -
            s[0] * static_cast<double>(h[2])));
  out[2] = static_cast<double>(gyro) *
      (static_cast<double>(h[2]) +
       static_cast<double>(alpha) *
           (s[0] * static_cast<double>(h[1]) -
            s[1] * static_cast<double>(h[0])));
}

JAMS_HOST_DEVICE inline void rodrigues_rotate(const double phi[3],
                                              const double s[3],
                                              double out[3]) {
  const double th2 = norm_squared3(phi);

  double c1[3];
  cross_product3(phi, s, c1);
  double c2[3];
  cross_product3(phi, c1, c2);

  if (th2 < 1e-8) {
    const double th4 = th2 * th2;
    const double a = 1.0 - th2 * (1.0 / 6.0) + th4 * (1.0 / 120.0);
    const double b = 0.5 - th2 * (1.0 / 24.0) + th4 * (1.0 / 720.0);

    for (int n = 0; n < 3; ++n) {
      out[n] = s[n] + a * c1[n] + b * c2[n];
    }
    return;
  }

  const double th = std::sqrt(th2);

  double sin_th;
  double cos_th;
  sincos_scalar(th, &sin_th, &cos_th);

  const double a = sin_th / th;
  const double b = (1.0 - cos_th) / th2;

  for (int n = 0; n < 3; ++n) {
    out[n] = s[n] + a * c1[n] + b * c2[n];
  }
}

JAMS_HOST_DEVICE inline void dexp_inv_so3(const double phi[3],
                                          const double v[3],
                                          double out[3]) {
  const double th2 = norm_squared3(phi);

  double c1[3];
  cross_product3(phi, v, c1);
  double c2[3];
  cross_product3(phi, c1, c2);

  if (th2 < 1e-8) {
    const double th4 = th2 * th2;
    const double beta =
        (1.0 / 12.0) + th2 * (1.0 / 720.0) + th4 * (1.0 / 30240.0);

    for (int n = 0; n < 3; ++n) {
      out[n] = v[n] - 0.5 * c1[n] + beta * c2[n];
    }
    return;
  }

  const double th = std::sqrt(th2);
  const double half = 0.5 * th;

  double sin_half;
  double cos_half;
  sincos_scalar(half, &sin_half, &cos_half);

  const double cot_half = cos_half / sin_half;
  const double beta = (1.0 / th2) * (1.0 - half * cot_half);

  for (int n = 0; n < 3; ++n) {
    out[n] = v[n] - 0.5 * c1[n] + beta * c2[n];
  }
}

template <typename NoiseReal, typename GyroReal, typename AlphaReal>
JAMS_HOST_DEVICE JAMS_FORCEINLINE void noise_step_rodrigues(
    const double s[3],
    const NoiseReal h_noise[3],
    const GyroReal gyro,
    const AlphaReal alpha,
    const double dt,
    double out[3]) {
  double omega[3];
  omega_llg(s, h_noise, gyro, alpha, omega);

  const double phi[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
  rodrigues_rotate(phi, s, out);
}

}  // namespace jams::solvers::rkmk

#endif  // JAMS_SOLVERS_LLG_RKMK_FUNCTIONS_H
