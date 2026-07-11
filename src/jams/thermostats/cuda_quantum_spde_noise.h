// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_QUANTUM_SPDE_NOISE_H
#define JAMS_CUDA_QUANTUM_SPDE_NOISE_H

#include "jams/thermostats/quantum_spde_noise.h"

#if HAS_CUDA

#include <array>
#include <cstddef>

#include <curand.h>

#include "jams/containers/multiarray.h"
#include "jams/cuda/cuda_stream.h"

namespace jams {

[[nodiscard]] int quantum_spde_cuda_process_count(int spin_count);
[[nodiscard]] std::size_t quantum_spde_cuda_curand_buffer_count(int process_count);
[[nodiscard]] int quantum_spde_cuda_grid_size(int process_count, int block_size);

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
  void initialize(Initialization initialization,
                  const jams::MultiArray<jams::Real, 1>& temperature);
  void initialize_stationary(jams::Real temperature);
  void initialize_stationary(const jams::MultiArray<jams::Real, 1>& temperature);
  void warmup(unsigned steps, jams::Real temperature, jams::Real* noise,
              const jams::Real* sigma);
  void warmup(unsigned steps, const jams::MultiArray<jams::Real, 1>& temperature,
              jams::Real* noise, const jams::Real* sigma);
  void update(jams::Real* noise, const jams::Real* sigma, jams::Real temperature);
  void update(jams::Real* noise, const jams::Real* sigma,
              const jams::MultiArray<jams::Real, 1>& temperature);
  void synchronize();

  [[nodiscard]] int process_count() const { return process_count_; }
  [[nodiscard]] bool zero_point_enabled() const { return zero_point_; }

  [[nodiscard]] const jams::MultiArray<double, 1>& zeta0_component(int component) const;
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta5() const { return zeta5_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta5p() const { return zeta5p_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta6() const { return zeta6_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta6p() const { return zeta6p_; }

 private:
  void generate_random_buffers();
  void prepare_fixed_temperature_coefficients(jams::Real temperature);
  void prepare_temperature_profile_coefficients(
      const jams::MultiArray<jams::Real, 1>& temperature);
  void zero_state();

  int process_count_ = 0;
  std::size_t curand_buffer_count_ = 0;
  double delta_tau_ = 0.0;
  double omega_max_ = 0.0;
  bool zero_point_ = false;
  bool fast_coefficients_valid_ = false;
  jams::Real fast_temperature_ = -1.0;
  QuantumSpdeBoseUpdateCoefficients fast_factor5_;
  QuantumSpdeBoseUpdateCoefficients fast_factor6_;
  QuantumSpdeZeroPointUpdateCoefficients zero_point_coefficients_;
  bool profile_has_positive_temperature_ = false;
  jams::MultiArray<QuantumSpdeBoseUpdateCoefficients, 1> profile_factor5_;
  jams::MultiArray<QuantumSpdeBoseUpdateCoefficients, 1> profile_factor6_;
  jams::MultiArray<QuantumSpdeBoseCholesky, 1> profile_stationary_factor5_;
  jams::MultiArray<QuantumSpdeBoseCholesky, 1> profile_stationary_factor6_;

  CudaStream& update_stream_;
  CudaStream curand_stream_{CudaStream::Priority::LOW};
  curandGenerator_t curand_generator_ = nullptr;
  cudaEvent_t curand_done_{};
  cudaEvent_t eta1a_reusable_{};
  cudaEvent_t eta1b_reusable_{};
  cudaEvent_t eta0a_reusable_{};
  cudaEvent_t eta0b_reusable_{};

  std::array<jams::MultiArray<double, 1>, 4> zeta0_;
  jams::MultiArray<double, 1> zeta5_;
  jams::MultiArray<double, 1> zeta5p_;
  jams::MultiArray<double, 1> zeta6_;
  jams::MultiArray<double, 1> zeta6p_;
  std::array<jams::MultiArray<jams::Real, 1>, 4> eta0a_;
  std::array<jams::MultiArray<jams::Real, 1>, 4> eta0b_;
  jams::MultiArray<jams::Real, 1> eta5a_;
  jams::MultiArray<jams::Real, 1> eta5b_;
  jams::MultiArray<jams::Real, 1> eta6a_;
  jams::MultiArray<jams::Real, 1> eta6b_;
};

}  // namespace jams

#endif  // HAS_CUDA
#endif  // JAMS_CUDA_QUANTUM_SPDE_NOISE_H
