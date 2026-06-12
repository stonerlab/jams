// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/quantum_spde_noise.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "jams/helpers/consts.h"

namespace {

void solve_3x3(double a[3][4]) {
  for (auto pivot = 0; pivot < 3; ++pivot) {
    auto pivot_row = pivot;
    auto pivot_abs = std::fabs(a[pivot][pivot]);
    for (auto row = pivot + 1; row < 3; ++row) {
      const auto row_abs = std::fabs(a[row][pivot]);
      if (row_abs > pivot_abs) {
        pivot_abs = row_abs;
        pivot_row = row;
      }
    }

    if (pivot_abs == 0.0) {
      throw std::runtime_error("singular stationary covariance system in quantum SPDE thermostat");
    }

    if (pivot_row != pivot) {
      for (auto col = pivot; col < 4; ++col) {
        std::swap(a[pivot][col], a[pivot_row][col]);
      }
    }

    const double inv_pivot = 1.0 / a[pivot][pivot];
    for (auto col = pivot; col < 4; ++col) {
      a[pivot][col] *= inv_pivot;
    }

    for (auto row = 0; row < 3; ++row) {
      if (row == pivot) {
        continue;
      }

      const double factor = a[row][pivot];
      for (auto col = pivot; col < 4; ++col) {
        a[row][col] -= factor * a[pivot][col];
      }
    }
  }
}

}  // namespace

namespace jams {

void quantum_spde_bose_exact_update_host(const double gamma, const double omega,
                                         const double eta0, const double h,
                                         double z[2]) {
  const double omega2 = omega * omega;
  const double alpha = 0.5 * gamma;
  const double decay = std::exp(-alpha * h);
  const double force_eq = eta0 / omega2;

  double y0 = z[0] - force_eq;
  double v0 = z[1];

  const double discriminant = omega2 - alpha * alpha;
  if (discriminant > 0.0) {
    const double beta = std::sqrt(discriminant);
    const double c = std::cos(beta * h);
    const double s = std::sin(beta * h);
    const double inv_beta = 1.0 / beta;

    const double y1 = decay * (y0 * c + (v0 + alpha * y0) * inv_beta * s);
    const double v1 = decay * (v0 * c - (alpha * v0 + omega2 * y0) * inv_beta * s);

    z[0] = y1 + force_eq;
    z[1] = v1;
    return;
  }

  if (discriminant < 0.0) {
    const double beta = std::sqrt(-discriminant);
    const double c = std::cosh(beta * h);
    const double s = std::sinh(beta * h);
    const double inv_beta = 1.0 / beta;

    const double y1 = decay * (y0 * c + (v0 + alpha * y0) * inv_beta * s);
    const double v1 = decay * (v0 * c - (alpha * v0 + omega2 * y0) * inv_beta * s);

    z[0] = y1 + force_eq;
    z[1] = v1;
    return;
  }

  const double y1 = decay * (y0 + (v0 + alpha * y0) * h);
  const double v1 = decay * (v0 - alpha * (v0 + alpha * y0) * h);

  z[0] = y1 + force_eq;
  z[1] = v1;
}

QuantumSpdeBoseUpdateCoefficients quantum_spde_bose_update_coefficients(
    const double gamma, const double omega, const double h) {
  if (h <= 0.0) {
    return {};
  }

  const double omega2 = omega * omega;
  const double alpha = 0.5 * gamma;
  const double decay = std::exp(-alpha * h);
  const double discriminant = omega2 - alpha * alpha;

  QuantumSpdeBoseUpdateCoefficients coeffs;
  coeffs.inv_omega2 = 1.0 / omega2;
  coeffs.eta_scale = static_cast<jams::Real>(std::sqrt(2.0 * gamma / h));

  if (discriminant > 0.0) {
    const double beta = std::sqrt(discriminant);
    const double c = std::cos(beta * h);
    const double s = std::sin(beta * h);
    const double inv_beta = 1.0 / beta;

    coeffs.m00 = decay * (c + alpha * inv_beta * s);
    coeffs.m01 = decay * inv_beta * s;
    coeffs.m10 = -decay * omega2 * inv_beta * s;
    coeffs.m11 = decay * (c - alpha * inv_beta * s);
  } else if (discriminant < 0.0) {
    const double beta = std::sqrt(-discriminant);
    const double c = std::cosh(beta * h);
    const double s = std::sinh(beta * h);
    const double inv_beta = 1.0 / beta;

    coeffs.m00 = decay * (c + alpha * inv_beta * s);
    coeffs.m01 = decay * inv_beta * s;
    coeffs.m10 = -decay * omega2 * inv_beta * s;
    coeffs.m11 = decay * (c - alpha * inv_beta * s);
  } else {
    coeffs.m00 = decay * (1.0 + alpha * h);
    coeffs.m01 = decay * h;
    coeffs.m10 = -decay * alpha * alpha * h;
    coeffs.m11 = decay * (1.0 - alpha * h);
  }

  coeffs.force0 = 1.0 - coeffs.m00;
  coeffs.force1 = -coeffs.m10;
  return coeffs;
}

QuantumSpdeZeroPointUpdateCoefficients quantum_spde_zero_point_update_coefficients(
    const double delta_tau, const double omega_max) {
  const double h_omega_max = (kHBarIU * omega_max * delta_tau) / kBoltzmannIU;
  if (h_omega_max <= 0.0) {
    return {};
  }

  QuantumSpdeZeroPointUpdateCoefficients coeffs;
  coeffs.zero_point_scale = static_cast<jams::Real>((kHBarIU * omega_max) / kBoltzmannIU);
  for (auto i = 0; i < 4; ++i) {
    const double lambda_h = kQuantumSpdeZeroPointLambdaFactors[i] * h_omega_max;
    coeffs.decay[i] = std::exp(-lambda_h);
    coeffs.eta_scale[i] = static_cast<jams::Real>(std::sqrt(2.0 / lambda_h));
    coeffs.weight[i] = static_cast<jams::Real>(kQuantumSpdeZeroPointWeights[i]);
  }
  return coeffs;
}

double quantum_spde_zero_point_stationary_variance(const double h_omega_max,
                                                   const int process_index) {
  const double lambda_h =
      kQuantumSpdeZeroPointLambdaFactors[process_index] * h_omega_max;
  if (lambda_h <= 0.0) {
    return 0.0;
  }
  const double decay = std::exp(-lambda_h);
  return 2.0 * (1.0 - decay) / (lambda_h * (1.0 + decay));
}

QuantumSpdeBoseCholesky quantum_spde_stationary_bose_cholesky(
    const double gamma, const double omega, const double h) {
  if (h <= 0.0) {
    return {};
  }

  double z[2] = {1.0, 0.0};
  quantum_spde_bose_exact_update_host(gamma, omega, 0.0, h, z);
  const double a00 = z[0];
  const double a10 = z[1];

  z[0] = 0.0;
  z[1] = 1.0;
  quantum_spde_bose_exact_update_host(gamma, omega, 0.0, h, z);
  const double a01 = z[0];
  const double a11 = z[1];

  z[0] = 0.0;
  z[1] = 0.0;
  quantum_spde_bose_exact_update_host(gamma, omega, 1.0, h, z);
  const double b0 = z[0];
  const double b1 = z[1];

  const double force_variance = 2.0 * gamma / h;
  const double q00 = force_variance * b0 * b0;
  const double q01 = force_variance * b0 * b1;
  const double q11 = force_variance * b1 * b1;

  double system[3][4] = {
      {1.0 - a00 * a00, -2.0 * a00 * a01, -a01 * a01, q00},
      {-a00 * a10, 1.0 - (a00 * a11 + a01 * a10), -a01 * a11, q01},
      {-a10 * a10, -2.0 * a10 * a11, 1.0 - a11 * a11, q11},
  };

  solve_3x3(system);

  const double p00 = system[0][3];
  const double p01 = system[1][3];
  const double p11 = system[2][3];
  if (p00 <= 0.0) {
    throw std::runtime_error("non-positive stationary position variance in quantum SPDE thermostat");
  }

  QuantumSpdeBoseCholesky factor;
  factor.l00 = std::sqrt(p00);
  factor.l10 = p01 / factor.l00;
  factor.l11 = std::sqrt(std::max(0.0, p11 - factor.l10 * factor.l10));
  return factor;
}

}  // namespace jams
