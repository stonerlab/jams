// Copyright 2014 Joseph Barker. All rights reserved.

// This noise generator is designed to reproduce a semiquantum thermostat
// which has a coth(omega) frequency dependence.

#ifndef JAMS_CUDA_QUANTUM_SPDE_NOISE_GENERATOR_H
#define JAMS_CUDA_QUANTUM_SPDE_NOISE_GENERATOR_H

#if HAS_CUDA

#include <curand.h>

#include "jams/core/noise_generator.h"
#include "jams/helpers/consts.h"

struct CudaQuantumSpdeNoiseGeneratorConfig {
  bool zero_point = false;
  double omega_max = 25.0 * kTwoPi;
};

class CudaQuantumSpdeNoiseGenerator : public NoiseGenerator {
 public:
  CudaQuantumSpdeNoiseGenerator(
      const jams::Real& temperature,
      const jams::Real timestep,
      int num_vectors,
      const CudaQuantumSpdeNoiseGeneratorConfig& config = {});
  ~CudaQuantumSpdeNoiseGenerator() override;

  void update() override;

  double stationary_variance() const;

 private:
  void initialize_stationary_state();

  static double no_zero_process_variance_component(double gamma, double omega, double step);

  CudaStream curand_stream_{CudaStream::Priority::LOW};
  cudaEvent_t curand_done_{};
  bool do_zero_point_ = false;

  jams::MultiArray<double, 1> zeta0_;
  jams::MultiArray<double, 1> zeta5_;
  jams::MultiArray<double, 1> zeta5p_;
  jams::MultiArray<double, 1> zeta6_;
  jams::MultiArray<double, 1> zeta6p_;
  jams::MultiArray<jams::Real, 1> eta0a_;
  jams::MultiArray<jams::Real, 1> eta0b_;
  jams::MultiArray<jams::Real, 1> eta1a_;
  jams::MultiArray<jams::Real, 1> eta1b_;
  double delta_tau_;
  double omega_max_;
  int num_channels_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_QUANTUM_SPDE_NOISE_GENERATOR_H
