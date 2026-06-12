// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_QUANTUM_SPDE_NOISE_H
#define JAMS_QUANTUM_SPDE_NOISE_H

#include <array>

#include "jams/helpers/mixed_precision.h"

namespace jams {

inline constexpr double kQuantumSpdeGamma5 = 5.0142;
inline constexpr double kQuantumSpdeOmega5 = 2.7189;
inline constexpr double kQuantumSpdeWeight5 = 1.8315;
inline constexpr double kQuantumSpdeGamma6 = 3.2974;
inline constexpr double kQuantumSpdeOmega6 = 1.2223;
inline constexpr double kQuantumSpdeWeight6 = 0.3429;
inline constexpr std::array<double, 4> kQuantumSpdeZeroPointWeights = {
    1.043576, 0.177222, 0.050319, 0.010241};
inline constexpr std::array<double, 4> kQuantumSpdeZeroPointLambdaFactors = {
    1.763817, 0.394613, 0.103506, 0.015873};

struct QuantumSpdeBoseCholesky {
  double l00 = 0.0;
  double l10 = 0.0;
  double l11 = 0.0;
};

struct QuantumSpdeBoseUpdateCoefficients {
  double m00 = 0.0;
  double m01 = 0.0;
  double m10 = 0.0;
  double m11 = 0.0;
  double force0 = 0.0;
  double force1 = 0.0;
  double inv_omega2 = 0.0;
  jams::Real eta_scale = 0.0;
};

struct QuantumSpdeZeroPointUpdateCoefficients {
  double decay[4] = {};
  jams::Real eta_scale[4] = {};
  jams::Real weight[4] = {};
  jams::Real zero_point_scale = 0.0;
};

void quantum_spde_bose_exact_update_host(double gamma, double omega, double eta0,
                                         double h, double z[2]);

QuantumSpdeBoseUpdateCoefficients quantum_spde_bose_update_coefficients(double gamma,
                                                                        double omega,
                                                                        double h);

QuantumSpdeZeroPointUpdateCoefficients quantum_spde_zero_point_update_coefficients(
    double delta_tau, double omega_max);

double quantum_spde_zero_point_stationary_variance(double h_omega_max, int process_index);

QuantumSpdeBoseCholesky quantum_spde_stationary_bose_cholesky(double gamma,
                                                              double omega,
                                                              double h);

}  // namespace jams

#endif  // JAMS_QUANTUM_SPDE_NOISE_H
