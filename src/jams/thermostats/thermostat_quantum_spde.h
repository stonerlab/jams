// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_THERMOSTAT_QUANTUM_SPDE_H
#define JAMS_THERMOSTAT_QUANTUM_SPDE_H

#include <memory>
#include <random>
#include <vector>

#include <pcg_random.hpp>

#include "jams/core/thermostat.h"
#include "jams/thermostats/quantum_spde_noise.h"

namespace jams {

class QuantumSpdeNoiseGenerator {
 public:
  enum class Initialization {
    Zero,
    Stationary,
  };

  QuantumSpdeNoiseGenerator(int process_count, double delta_tau,
                            double omega_max, bool zero_point);

  QuantumSpdeNoiseGenerator(const QuantumSpdeNoiseGenerator&) = delete;
  QuantumSpdeNoiseGenerator& operator=(const QuantumSpdeNoiseGenerator&) = delete;

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

  [[nodiscard]] int process_count() const { return process_count_; }
  [[nodiscard]] bool zero_point_enabled() const { return zero_point_; }

  [[nodiscard]] const jams::MultiArray<double, 1>& zeta0() const { return zeta0_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta5() const { return zeta5_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta5p() const { return zeta5p_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta6() const { return zeta6_; }
  [[nodiscard]] const jams::MultiArray<double, 1>& zeta6p() const { return zeta6p_; }

 private:
  void ensure_random_generators(int count);
  void fill_standard_normal(jams::MultiArray<jams::Real, 1>& values);
  void prepare_fixed_temperature_coefficients(jams::Real temperature);
  void prepare_temperature_profile_coefficients(
      const jams::MultiArray<jams::Real, 1>& temperature);
  void update_zero_point(jams::Real* noise, const jams::Real* sigma);
  void zero_noise(jams::Real* noise) const;
  void zero_state();

  int process_count_ = 0;
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

  pcg32_k1024 random_generator_;
  std::vector<pcg32_k1024> random_generators_;

  jams::MultiArray<double, 1> zeta0_;
  jams::MultiArray<double, 1> zeta5_;
  jams::MultiArray<double, 1> zeta5p_;
  jams::MultiArray<double, 1> zeta6_;
  jams::MultiArray<double, 1> zeta6p_;
  jams::MultiArray<jams::Real, 1> eta0_;
  jams::MultiArray<jams::Real, 1> eta1_;
  jams::MultiArray<jams::Real, 1> eta_stationary_;
};

}  // namespace jams

class ThermostatQuantumSpde : public Thermostat {
 public:
  ThermostatQuantumSpde(const jams::Real& temperature,
                        const jams::Real& sigma,
                        jams::Real timestep,
                        int num_spins);

  void update() override;

 private:
  std::unique_ptr<jams::QuantumSpdeNoiseGenerator> noise_generator_;
  jams::MultiArray<jams::Real, 1> process_temperature_;
};

#endif  // JAMS_THERMOSTAT_QUANTUM_SPDE_H
