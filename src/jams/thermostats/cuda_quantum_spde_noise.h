// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_QUANTUM_SPDE_NOISE_H
#define JAMS_CUDA_QUANTUM_SPDE_NOISE_H

#if HAS_CUDA

#include <curand.h>

#include "jams/containers/multiarray.h"
#include "jams/cuda/cuda_stream.h"
#include "jams/helpers/mixed_precision.h"

namespace jams {

struct QuantumSpdeBoseCholesky {
  double l00 = 0.0;
  double l10 = 0.0;
  double l11 = 0.0;
};

void quantum_spde_bose_exact_update_host(double gamma, double omega, double eta0,
                                         double h, double z[2]);

QuantumSpdeBoseCholesky quantum_spde_stationary_bose_cholesky(double gamma,
                                                              double omega,
                                                              double h);

class CudaQuantumSpdeNoiseGenerator {
 public:
  enum class Initialization {
    Zero,
    Stationary,
  };

  CudaQuantumSpdeNoiseGenerator(int process_count, double delta_tau,
                                double omega_max, bool zero_point,
                                CudaStream& update_stream);
  ~CudaQuantumSpdeNoiseGenerator();

  CudaQuantumSpdeNoiseGenerator(const CudaQuantumSpdeNoiseGenerator&) = delete;
  CudaQuantumSpdeNoiseGenerator& operator=(const CudaQuantumSpdeNoiseGenerator&) = delete;

  void initialize(Initialization initialization, jams::Real temperature);
  void initialize_stationary(jams::Real temperature);
  void warmup(unsigned steps, jams::Real temperature, jams::Real* noise,
              const jams::Real* sigma);
  void update(jams::Real* noise, const jams::Real* sigma, jams::Real temperature);
  void synchronize();

  [[nodiscard]] int process_count() const { return process_count_; }
  [[nodiscard]] bool zero_point_enabled() const { return zero_point_; }

  [[nodiscard]] const jams::MultiArray<double, 1>& zeta0() const { return zeta0_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta5() const { return zeta5_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta5p() const { return zeta5p_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta6() const { return zeta6_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta6p() const { return zeta6p_; }

 private:
  void generate_random_buffers();
  void zero_state();

  int process_count_ = 0;
  double delta_tau_ = 0.0;
  double omega_max_ = 0.0;
  bool zero_point_ = false;

  CudaStream& update_stream_;
  CudaStream curand_stream_{CudaStream::Priority::LOW};
  cudaEvent_t curand_done_{};
  cudaEvent_t eta1a_reusable_{};
  cudaEvent_t eta1b_reusable_{};
  cudaEvent_t eta0a_reusable_{};
  cudaEvent_t eta0b_reusable_{};

  jams::MultiArray<double, 1> zeta0_;
  jams::MultiArray<double, 1> zeta5_;
  jams::MultiArray<double, 1> zeta5p_;
  jams::MultiArray<double, 1> zeta6_;
  jams::MultiArray<double, 1> zeta6p_;
  jams::MultiArray<jams::Real, 1> eta0a_;
  jams::MultiArray<jams::Real, 1> eta0b_;
  jams::MultiArray<jams::Real, 1> eta1a_;
  jams::MultiArray<jams::Real, 1> eta1b_;
};

}  // namespace jams

#endif  // HAS_CUDA
#endif  // JAMS_CUDA_QUANTUM_SPDE_NOISE_H
