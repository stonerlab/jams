#ifndef JAMS_CUDA_LLG_RS2_H
#define JAMS_CUDA_LLG_RS2_H

#if HAS_CUDA

#include <memory>
#include <type_traits>

#include <curand.h>

#include "jams/containers/multiarray.h"
#include "jams/cuda/cuda_solver.h"

class CudaThermostatQuantumSpde;

class CUDALLGRS2Solver : public CudaSolver {
 public:
  explicit CUDALLGRS2Solver(const libconfig::Setting& settings) { initialize(settings); }

  void initialize(const libconfig::Setting& settings) override;
  void run() override;

  std::string name() const override { return "llg-rs2-gpu"; }

  bool supports_deterministic_spin_rate() const override { return true; }
  double max_deterministic_spin_rate() override { return max_llg_spin_rate_cuda(gyro_eff_); }

 private:
  friend class CudaRS2TestAccess;

  void run_step();

  template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
  void run_with_parameters(GyroParam gyro, AlphaParam alpha, FieldScaleParam field_scale);

  jams::MultiArray<jams::Real, 1> gyro_eff_;
  bool gyro_eff_is_uniform_ = false;
  jams::Real gyro_eff_uniform_value_ = 0.0;
  bool alpha_is_uniform_ = false;
  jams::Real alpha_uniform_value_ = 0.0;
  bool mus_is_uniform_ = false;
  jams::Real mus_uniform_inv_value_ = 0.0;

  jams::MultiArray<double, 2> spins_initial_;
  jams::MultiArray<double, 2> f1_;
  jams::MultiArray<double, 2> f2_;
  jams::MultiArray<double, 2> b_;

  bool has_coloured_bath_ = false;
  bool coloured_sample_ready_ = false;
  CudaThermostatQuantumSpde* spde_thermostat_ = nullptr;
  double white_temperature_ = 0.0;
  jams::MultiArray<double, 1> white_sigma_;
  jams::MultiArray<double, 2> uniforms_;
  std::unique_ptr<std::remove_pointer_t<curandGenerator_t>, decltype(&curandDestroyGenerator)>
      white_generator_{nullptr, &curandDestroyGenerator};
};

#endif  // HAS_CUDA

#endif  // JAMS_CUDA_LLG_RS2_H
