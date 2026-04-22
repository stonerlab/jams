// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CPU_GENERAL_FFT_NOISE_GENERATOR_H
#define JAMS_CPU_GENERAL_FFT_NOISE_GENERATOR_H

#include <functional>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "jams/core/noise_generator.h"

struct CpuGeneralFFTNoiseGeneratorConfig {
  std::string spectrum = "quantum-lorentzian";
  double lorentzian_omega0 = std::numeric_limits<double>::quiet_NaN();
  double lorentzian_gamma = std::numeric_limits<double>::quiet_NaN();
  bool write_diagnostics = false;
};

class CpuGeneralFFTNoiseGenerator : public NoiseGenerator {
 public:
  CpuGeneralFFTNoiseGenerator(const jams::Real& temperature,
                              jams::Real timestep,
                              int num_vectors,
                              const CpuGeneralFFTNoiseGeneratorConfig& config = {});
  ~CpuGeneralFFTNoiseGenerator() override;

  void update() override;

  double stationary_variance() const;
  int memory_depth() const { return num_trunc_; }

 private:
  void init_lorentzian(const CpuGeneralFFTNoiseGeneratorConfig& config);

  static double classical_ohmic_spectrum(double omega, double temperature);
  static double quantum_ohmic_spectrum(double omega, double temperature);
  static double quantum_no_zero_ohmic_spectrum(double omega, double temperature);

  static double classical_lorentzian_spectrum(
      double omega, double temperature, double omega0, double gamma, double A);
  static double quantum_lorentzian_spectrum(
      double omega, double temperature, double omega0, double gamma, double A);
  static double quantum_no_zero_lorentzian_spectrum(
      double omega, double temperature, double omega0, double gamma, double A);

  void output_diagnostics(const std::vector<double>& discrete_psd,
                          const std::vector<double>& full_memory_kernel) const;
  void output_thermostat_properties(std::ostream& os) const;

  template<typename... Args>
  using SpectralFunctionSignature = std::function<double(double, Args...)>;

  template<typename... Args>
  static std::vector<double> discrete_sqrt_psd(
      SpectralFunctionSignature<Args...> spectral_function,
      const double& delta_omega,
      const int& num_freq,
      Args&&... args);

  static std::vector<double> discrete_real_fourier_transform(std::vector<double> x);

  std::ofstream debug_file_;

  int num_freq_ = 0;
  int num_trunc_ = 0;
  int iteration_ = 0;
  double max_omega_ = 0.0;
  double delta_omega_ = 0.0;
  double delta_t_ = 0.0;
  double filter_temperature_ = 0.0;

  int num_channels_even_ = 0;

  double lorentzian_omega0_ = 0.0;
  double lorentzian_gamma_ = 0.0;
  double lorentzian_A_ = 0.0;

  jams::MultiArray<double, 1> memory_kernel_;
  jams::MultiArray<double, 1> white_noise_;
};

template<typename... Args>
std::vector<double> CpuGeneralFFTNoiseGenerator::discrete_sqrt_psd(
    SpectralFunctionSignature<Args...> spectral_function,
    const double& delta_omega,
    const int& num_freq,
    Args&&... args) {
  std::vector<double> result(num_freq);

  for (auto i = 0; i < static_cast<int>(result.size()); ++i) {
    const double omega = i * delta_omega;
    result[i] = std::sqrt(spectral_function(omega, args...));
  }

  return result;
}

#endif  // JAMS_CPU_GENERAL_FFT_NOISE_GENERATOR_H
