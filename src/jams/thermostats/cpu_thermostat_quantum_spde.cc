// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/cpu_thermostat_quantum_spde.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include <jams/common.h>

#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"

namespace {

constexpr std::array<double, 4> kZeroPointC = {
    1.043576,
    0.177222,
    0.050319,
    0.010241,
};

constexpr std::array<double, 4> kZeroPointLambda = {
    1.763817,
    0.394613,
    0.103506,
    0.015873,
};

double ou_linear_update(const double z,
                        const double lambda,
                        const double eta,
                        const double h) {
  const double decay = std::exp(-lambda * h);
  return eta + (z - eta) * decay;
}

void bose_exact_update_host(const double gamma,
                            const double omega,
                            const double eta0,
                            const double h,
                            double z[2]) {
  const double omega2 = omega * omega;
  const double alpha = 0.5 * gamma;
  const double decay = std::exp(-alpha * h);
  const double force_eq = eta0 / omega2;

  const double y0 = z[0] - force_eq;
  const double v0 = z[1];

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

std::array<double, 3> solve_linear_system_3x3(
    const std::array<std::array<double, 4>, 3>& augmented) {
  auto a = augmented;
  for (int pivot = 0; pivot < 3; ++pivot) {
    int pivot_row = pivot;
    for (int row = pivot + 1; row < 3; ++row) {
      if (std::abs(a[row][pivot]) > std::abs(a[pivot_row][pivot])) {
        pivot_row = row;
      }
    }

    std::swap(a[pivot], a[pivot_row]);
    const double inv_pivot = 1.0 / a[pivot][pivot];
    for (int col = pivot; col < 4; ++col) {
      a[pivot][col] *= inv_pivot;
    }

    for (int row = 0; row < 3; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = a[row][pivot];
      for (int col = pivot; col < 4; ++col) {
        a[row][col] -= factor * a[pivot][col];
      }
    }
  }

  return {a[0][3], a[1][3], a[2][3]};
}

std::array<double, 3> stationary_oscillator_covariance(const double gamma,
                                                       const double omega,
                                                       const double h) {
  const double eta_scale = std::sqrt(2.0 * gamma / h);

  double basis0[2] = {1.0, 0.0};
  bose_exact_update_host(gamma, omega, 0.0, h, basis0);
  double basis1[2] = {0.0, 1.0};
  bose_exact_update_host(gamma, omega, 0.0, h, basis1);
  double noise[2] = {0.0, 0.0};
  bose_exact_update_host(gamma, omega, eta_scale, h, noise);

  const double a = basis0[0];
  const double b = basis1[0];
  const double c = basis0[1];
  const double d = basis1[1];

  const double q00 = noise[0] * noise[0];
  const double q01 = noise[0] * noise[1];
  const double q11 = noise[1] * noise[1];

  const std::array<std::array<double, 4>, 3> system = {{
      {{1.0 - a * a, -2.0 * a * b, -b * b, q00}},
      {{-a * c, 1.0 - a * d - b * c, -b * d, q01}},
      {{-c * c, -2.0 * c * d, 1.0 - d * d, q11}},
  }};

  return solve_linear_system_3x3(system);
}

double stationary_first_coordinate_variance(const double gamma,
                                            const double omega,
                                            const double h) {
  return stationary_oscillator_covariance(gamma, omega, h)[0];
}

double stationary_ou_variance(const double lambda, const double h) {
  const double x = lambda * h;
  if (x == 0.0) {
    return 1.0;
  }
  return (2.0 * std::tanh(0.5 * x)) / x;
}

std::array<double, 2> sample_bivariate_gaussian(
    const std::array<double, 3>& covariance,
    std::normal_distribution<double>& normal_distribution,
    jams::RandomGeneratorType& random_generator) {
  const double variance0 = std::max(0.0, covariance[0]);
  const double covariance01 = covariance[1];
  const double variance1 = std::max(0.0, covariance[2]);

  const double xi0 = normal_distribution(random_generator);
  const double xi1 = normal_distribution(random_generator);
  const double stddev0 = std::sqrt(variance0);
  const double conditional_mean_factor = stddev0 > 0.0 ? covariance01 / stddev0 : 0.0;
  const double conditional_variance =
      std::max(0.0, variance1 - covariance01 * covariance01 / std::max(variance0, 1.0e-30));

  return {
      stddev0 * xi0,
      conditional_mean_factor * xi0 + std::sqrt(conditional_variance) * xi1,
  };
}

}  // namespace

CpuQuantumSpdeNoiseGenerator::CpuQuantumSpdeNoiseGenerator(
    const jams::Real& temperature,
    const jams::Real timestep,
    const int num_vectors,
    const CpuQuantumSpdeNoiseGeneratorConfig& config)
    : NoiseGenerator(temperature, num_vectors),
      do_zero_point_(config.zero_point),
      delta_tau_((timestep * kBoltzmannIU) / kHBarIU),
      omega_max_(config.omega_max),
      num_channels_(3 * num_vectors) {
  std::cout << "\n  initialising quantum-spde-cpu noise generator\n";

  zeta5_.resize(num_channels_).zero();
  zeta5p_.resize(num_channels_).zero();
  zeta6_.resize(num_channels_).zero();
  zeta6p_.resize(num_channels_).zero();

  if (do_zero_point_) {
    zeta0_.resize(4 * num_channels_).zero();
  }

  std::cout << "    omega_max (THz) " << omega_max_ / kTwoPi << "\n";
  std::cout << "    hbar*w/kB " << (kHBarIU * omega_max_) / kBoltzmannIU << "\n";
  std::cout << "    t_step " << timestep << "\n";
  std::cout << "    delta tau " << delta_tau_ << "\n";

  initialize_stationary_state();
}

void CpuQuantumSpdeNoiseGenerator::update() {
  if (this->temperature() == 0) {
    noise_.zero();
    return;
  }

  const double reduced_omega_max =
      (kHBarIU * omega_max_) / (kBoltzmannIU * this->temperature());
  const double reduced_delta_tau = delta_tau_ * this->temperature();
  const double temperature = this->temperature();

  std::normal_distribution<double> normal_distribution;
  auto& random_generator = jams::instance().random_generator();

  for (int channel = 0; channel < num_channels_; ++channel) {
    double s1 = 0.0;

    double z5[2] = {zeta5_(channel), zeta5p_(channel)};
    const double eta5 =
        normal_distribution(random_generator) * std::sqrt(2.0 * 5.0142 / reduced_delta_tau);
    bose_exact_update_host(5.0142, 2.7189, eta5, reduced_delta_tau, z5);
    zeta5_(channel) = z5[0];
    zeta5p_(channel) = z5[1];
    s1 += 1.8315 * z5[0];

    double z6[2] = {zeta6_(channel), zeta6p_(channel)};
    const double eta6 =
        normal_distribution(random_generator) * std::sqrt(2.0 * 3.2974 / reduced_delta_tau);
    bose_exact_update_host(3.2974, 1.2223, eta6, reduced_delta_tau, z6);
    zeta6_(channel) = z6[0];
    zeta6p_(channel) = z6[1];
    s1 += 0.3429 * z6[0];

    double noise_value = temperature * s1;

    if (do_zero_point_) {
      double s0 = 0.0;
      for (int mode = 0; mode < static_cast<int>(kZeroPointC.size()); ++mode) {
        const double lambda = kZeroPointLambda[mode] * reduced_omega_max;
        const double c = kZeroPointC[mode] * reduced_omega_max;
        const double eta = normal_distribution(random_generator)
            * std::sqrt(2.0 / (lambda * reduced_delta_tau));
        const int index = 4 * channel + mode;
        const double updated = ou_linear_update(zeta0_(index), lambda, eta, reduced_delta_tau);
        zeta0_(index) = updated;
        s0 += c * (eta - updated);
      }
      noise_value += temperature * s0;
    }

    noise_(channel / 3, channel % 3) = static_cast<jams::Real>(noise_value);
  }
}

double CpuQuantumSpdeNoiseGenerator::stationary_variance() const {
  if (this->temperature() == 0) {
    return 0.0;
  }
  if (do_zero_point_) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double h = delta_tau_ * this->temperature();
  const double var5 = no_zero_process_variance_component(5.0142, 2.7189, h);
  const double var6 = no_zero_process_variance_component(3.2974, 1.2223, h);
  return pow2(this->temperature())
         * (pow2(1.8315) * var5 + pow2(0.3429) * var6);
}

void CpuQuantumSpdeNoiseGenerator::initialize_stationary_state() {
  if (this->temperature() == 0) {
    return;
  }

  const double reduced_delta_tau = delta_tau_ * this->temperature();
  const double reduced_omega_max =
      (kHBarIU * omega_max_) / (kBoltzmannIU * this->temperature());

  auto& random_generator = jams::instance().random_generator();
  std::normal_distribution<double> normal_distribution;

  const auto initialize_oscillator_pair =
      [&](jams::MultiArray<double, 1>& zeta,
          jams::MultiArray<double, 1>& zeta_prime,
          const double gamma,
          const double omega) {
        const auto covariance =
            stationary_oscillator_covariance(gamma, omega, reduced_delta_tau);
        for (int i = 0; i < num_channels_; ++i) {
          const auto sample =
              sample_bivariate_gaussian(covariance, normal_distribution, random_generator);
          zeta(i) = sample[0];
          zeta_prime(i) = sample[1];
        }
      };

  initialize_oscillator_pair(zeta5_, zeta5p_, 5.0142, 2.7189);
  initialize_oscillator_pair(zeta6_, zeta6p_, 3.2974, 1.2223);

  if (!do_zero_point_) {
    return;
  }

  for (int channel = 0; channel < num_channels_; ++channel) {
    for (int mode = 0; mode < static_cast<int>(kZeroPointLambda.size()); ++mode) {
      const double lambda = kZeroPointLambda[mode] * reduced_omega_max;
      const double variance = stationary_ou_variance(lambda, reduced_delta_tau);
      zeta0_(4 * channel + mode) = std::sqrt(variance) * normal_distribution(random_generator);
    }
  }
}

double CpuQuantumSpdeNoiseGenerator::no_zero_process_variance_component(
    const double gamma,
    const double omega,
    const double step) {
  return stationary_first_coordinate_variance(gamma, omega, step);
}
