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

JAMS_HOST_DEVICE JAMS_FORCEINLINE double norm3(const double v[3]) {
#if defined(__CUDA_ARCH__)
  return norm3d(v[0], v[1], v[2]);
#else
  return std::hypot(v[0], v[1], v[2]);
#endif
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

  // Defining Rodrigues rotation of S around a vector φ
  //    R(φ) S = S + (sin(|φ|)/|φ|) (φ x S) + ((1 - cos |φ|) / |φ|^2) (φ x (φ x S))
  //
  // There are two issues:
  // 1. (1 - cos|φ|)/|φ|^2 suffers catastrophic cancellation as |φ| -> 0
  // 2. (sin(|φ|)/|φ|) is undefined at |φ| = 0
  //
  // We can fix 1. using the identity
  //   (1 - cos x)/x^2 = (1/2)( sin(x/2) / (x/2) )^2 = (1/2) sinc^2(x/2)
  //
  // We now need to calculate sin(x)/x for both x=|φ| and x=≈. Actually we can simplify
  // using the half angle identity
  //    sin(x) / x = cos(x/2) sin(x/2) / (x/2)
  //
  // hence we can work purely in terms of |φ| and make use of the CUDA intrinsic sincos function.
  //
  // Thus we rewrite Rodrigues formula as
  //    R(φ) S = S + sinc(|φ|/2) cos(|φ|/2) (φ x S) + (1/2) sinc^2(|φ|/2) (φ x (φ x S))
  //
  // Now all we need to do is avoid the potential sin(0) / 0. This is only an issue at exactly
  // |φ|/2 = 0, so we have a simple guard for that. In a real spin dynamics calculation that case

  const double half = 0.5 * norm3(phi);

  double sin_half, cos_half;
  sincos_scalar(half, &sin_half, &cos_half);

  // By definition sinc(0)/0 = 1. Use denom to selection to avoid division by zero.
  const double denom = (half == 0.0) ? 1.0 : half;
  const double sinc_half = sin_half / denom;

  const double a = sinc_half * cos_half;
  const double b = 0.5 * sinc_half * sinc_half;

  double c1[3];
  cross_product3(phi, s, c1);
  double c2[3];
  cross_product3(phi, c1, c2);

  for (int n = 0; n < 3; ++n) {
    out[n] = s[n] + a * c1[n] + b * c2[n];
  }
}

  JAMS_HOST_DEVICE inline void dexp_inv_so3(const double phi[3],
                                           const double v[3],
                                           double out[3]) {
  const double th2 = norm_squared3(phi);

  double beta;
  if (th2 < 1e-3) {
    // Taylor series through theta^4, including beta(0) = 1/12.
    beta = 1.0 / 12.0 + th2 * (1.0 / 720.0 + th2 * (1.0 / 30240.0));
  } else {
    const double half = 0.5 * std::sqrt(th2);

    double sin_half, cos_half;
    sincos_scalar(half, &sin_half, &cos_half);

    // beta = (sin(h) - h*cos(h)) / (theta^2*sin(h)).
    // The inverse differential is singular at nonzero multiples of 2*pi.
    beta = std::fma(-half, cos_half, sin_half) / (th2 * sin_half);
  }

  double c1[3];
  cross_product3(phi, v, c1);
  double c2[3];
  cross_product3(phi, c1, c2);

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
