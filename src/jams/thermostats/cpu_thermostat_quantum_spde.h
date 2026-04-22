// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CPU_QUANTUM_SPDE_NOISE_GENERATOR_H
#define JAMS_CPU_QUANTUM_SPDE_NOISE_GENERATOR_H

#include "jams/core/noise_generator.h"
#include "jams/helpers/consts.h"

struct CpuQuantumSpdeNoiseGeneratorConfig {
  bool zero_point = false;
  double omega_max = 25.0 * kTwoPi;
};

class CpuQuantumSpdeNoiseGenerator : public NoiseGenerator {
 public:
  CpuQuantumSpdeNoiseGenerator(
      const jams::Real& temperature,
      const jams::Real timestep,
      int num_vectors,
      const CpuQuantumSpdeNoiseGeneratorConfig& config = {});

  void update() override;

  double stationary_variance() const;

 private:
  void initialize_stationary_state();

  static double no_zero_process_variance_component(double gamma, double omega, double step);

  bool do_zero_point_ = false;
  jams::MultiArray<double, 1> zeta0_;
  jams::MultiArray<double, 1> zeta5_;
  jams::MultiArray<double, 1> zeta5p_;
  jams::MultiArray<double, 1> zeta6_;
  jams::MultiArray<double, 1> zeta6p_;
  double delta_tau_ = 0.0;
  double omega_max_ = 0.0;
  int num_channels_ = 0;
};

#endif  // JAMS_CPU_QUANTUM_SPDE_NOISE_GENERATOR_H
