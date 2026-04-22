// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/cpu_thermostat_general_fft.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <jams/common.h>

#include "jams/helpers/consts.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/output.h"
#include "jams/helpers/utils.h"
#include "jams/interface/fft.h"
#include "jams/maths/functions.h"

namespace {

int periodic_index(const int i, const int size) {
  return (i + size) % size;
}

}  // namespace

CpuGeneralFFTNoiseGenerator::CpuGeneralFFTNoiseGenerator(
    const jams::Real& temperature,
    const jams::Real timestep,
    const int num_vectors,
    const CpuGeneralFFTNoiseGeneratorConfig& config)
    : NoiseGenerator(temperature, num_vectors),
      filter_temperature_(temperature) {
  std::cout << "\n  initialising general-fft-cpu noise generator\n";

  const auto noise_spectrum_type = lowercase(config.spectrum);

  delta_t_ = timestep;
  max_omega_ = kPi / delta_t_;

  std::function<double(double)> psd_function;
  if (noise_spectrum_type == "classical-ohmic" || noise_spectrum_type == "classical") {
    psd_function = [&](const double omega) {
      return classical_ohmic_spectrum(omega, filter_temperature_);
    };
  } else if (noise_spectrum_type == "quantum-ohmic") {
    psd_function = [&](const double omega) {
      return quantum_ohmic_spectrum(omega, filter_temperature_);
    };
  } else if (noise_spectrum_type == "quantum-no-zero-ohmic"
             || noise_spectrum_type == "bose-einstein") {
    psd_function = [&](const double omega) {
      return quantum_no_zero_ohmic_spectrum(omega, filter_temperature_);
    };
  } else if (noise_spectrum_type == "classical-lorentzian") {
    init_lorentzian(config);
    psd_function = [&](const double omega) {
      return classical_lorentzian_spectrum(
          omega, filter_temperature_, lorentzian_omega0_, lorentzian_gamma_, lorentzian_A_);
    };
  } else if (noise_spectrum_type == "quantum-lorentzian") {
    init_lorentzian(config);
    psd_function = [&](const double omega) {
      return quantum_lorentzian_spectrum(
          omega, filter_temperature_, lorentzian_omega0_, lorentzian_gamma_, lorentzian_A_);
    };
  } else if (noise_spectrum_type == "quantum-no-zero-lorentzian"
             || noise_spectrum_type == "no-zero-quantum-lorentzian") {
    init_lorentzian(config);
    psd_function = [&](const double omega) {
      return quantum_no_zero_lorentzian_spectrum(
          omega, filter_temperature_, lorentzian_omega0_, lorentzian_gamma_, lorentzian_A_);
    };
  } else {
    throw std::runtime_error("unknown spectrum type '" + noise_spectrum_type + "'");
  }

  const int num_freq_init = 10000;
  const int num_freq_increment = 10000;
  const int num_freq_increment_attempts = 499;
  const double memory_function_equality_tolerance = 1e-7;

  num_freq_ = num_freq_init;
  delta_omega_ = max_omega_ / double(num_freq_);
  auto discrete_psd = discrete_sqrt_psd(psd_function, delta_omega_, num_freq_);
  auto memory_kernel = discrete_real_fourier_transform(discrete_psd);

  for (auto n = 0; n < num_freq_increment_attempts; ++n) {
    const int trial_num_freq = num_freq_ + num_freq_increment;
    auto trial_psd = discrete_sqrt_psd(
        psd_function, max_omega_ / double(trial_num_freq), trial_num_freq);
    auto trial_kernel = discrete_real_fourier_transform(trial_psd);

    const bool kernels_are_equal = std::equal(
        memory_kernel.begin(),
        memory_kernel.end(),
        trial_kernel.begin(),
        [&](const auto& x, const auto& y) {
          return approximately_equal(x, y, memory_function_equality_tolerance);
        });

    if (kernels_are_equal) {
      break;
    }

    num_freq_ = trial_num_freq;
    discrete_psd = std::move(trial_psd);
    memory_kernel = std::move(trial_kernel);
  }

  delta_omega_ = max_omega_ / double(num_freq_);

  const auto memory_absmax = *std::max_element(
      memory_kernel.begin(),
      memory_kernel.end(),
      [](const auto& a, const auto& b) {
        return std::abs(a) < std::abs(b);
      });

  const int trunc_block_size = 100;
  const double trunc_zero_tolerance = std::abs(memory_absmax) * 0.005;

  std::vector<double> block_averages;
  for (size_t i = 0; i < memory_kernel.size(); i += trunc_block_size) {
    const auto block_end = std::min(i + trunc_block_size, memory_kernel.size());
    const auto block_sum = std::accumulate(memory_kernel.begin() + i, memory_kernel.begin() + block_end, 0.0);
    block_averages.push_back(block_sum / double(block_end - i));
  }

  num_trunc_ = static_cast<int>(memory_kernel.size());
  for (auto i = 1U; i < block_averages.size(); ++i) {
    if (approximately_zero(block_averages[i], trunc_zero_tolerance)
        && approximately_zero(block_averages[i - 1], trunc_zero_tolerance)) {
      num_trunc_ = trunc_block_size * static_cast<int>(i);
      break;
    }
  }

  if (num_trunc_ <= 0) {
    throw jams::GeneralException("General FFT noise generator produced an empty memory kernel");
  }

  memory_kernel_.resize(num_trunc_);
  for (int i = 0; i < num_trunc_; ++i) {
    memory_kernel_(i) = std::sqrt(1.0 / delta_t_) * memory_kernel[i];
  }

  num_channels_even_ = noise_.elements() + (noise_.elements() % 2);
  white_noise_.resize(num_channels_even_ * (2 * num_trunc_ + 1));

  std::normal_distribution<double> normal_distribution;
  auto& random_generator = jams::instance().random_generator();
  for (auto i = 0; i < white_noise_.elements(); ++i) {
    white_noise_(i) = normal_distribution(random_generator);
  }

  if (config.write_diagnostics) {
    output_diagnostics(discrete_psd, memory_kernel);
  }

  output_thermostat_properties(std::cout);
}

CpuGeneralFFTNoiseGenerator::~CpuGeneralFFTNoiseGenerator() {
#ifdef PRINT_NOISE
  debug_file_.close();
#endif
}

void CpuGeneralFFTNoiseGenerator::update() {
  if (memory_kernel_.size() == 0) {
    throw std::runtime_error("General FFT noise generator memory kernel is empty");
  }

  if (filter_temperature_ != this->temperature()) {
    throw std::runtime_error(
        "you cannot dynamically change the temperature for the CpuGeneralFFTNoiseGenerator");
  }

  const int n = periodic_index(iteration_, 2 * num_trunc_ + 1);
  for (auto idx = 0; idx < noise_.elements(); ++idx) {
    double sum = memory_kernel_(0) * white_noise_(num_channels_even_ * n + idx);

    for (auto k = 1; k < num_trunc_; ++k) {
      const auto j_minus_k = periodic_index(n - k, 2 * num_trunc_ + 1);
      const auto j_plus_k = periodic_index(n + k, 2 * num_trunc_ + 1);
      sum += memory_kernel_(k) * (
          white_noise_(num_channels_even_ * j_plus_k + idx)
          + white_noise_(num_channels_even_ * j_minus_k + idx));
    }

    noise_.data()[idx] = static_cast<jams::Real>(sum);
  }

  std::normal_distribution<double> normal_distribution;
  auto& random_generator = jams::instance().random_generator();
  const auto start_index =
      num_channels_even_ * periodic_index(iteration_ + num_trunc_ + 1, 2 * num_trunc_ + 1);
  for (auto idx = 0; idx < num_channels_even_; ++idx) {
    white_noise_(start_index + idx) = normal_distribution(random_generator);
  }

  ++iteration_;
}

double CpuGeneralFFTNoiseGenerator::stationary_variance() const {
  if (num_trunc_ == 0) {
    return 0.0;
  }

  double variance = memory_kernel_(0) * memory_kernel_(0);
  for (int i = 1; i < num_trunc_; ++i) {
    variance += 2.0 * memory_kernel_(i) * memory_kernel_(i);
  }
  return variance;
}

void CpuGeneralFFTNoiseGenerator::output_diagnostics(
    const std::vector<double>& discrete_psd,
    const std::vector<double>& full_memory_kernel) const {
  std::ofstream noise_target_file(jams::output::full_path_filename("noise_target_spectrum.tsv"));
  noise_target_file << "freq_THz    spectrum_meV" << std::endl;
  for (auto i = 0U; i < discrete_psd.size(); ++i) {
    noise_target_file << jams::fmt::decimal << i * delta_omega_ / kTwoPi << " ";
    noise_target_file << jams::fmt::sci << discrete_psd[i] << "\n";
  }
  noise_target_file.close();

  std::ofstream filter_full_file(jams::output::full_path_filename("noise_filter_full.tsv"));
  filter_full_file << "delta_t_ps    filter_arb" << std::endl;
  for (auto i = 0U; i < full_memory_kernel.size(); ++i) {
    filter_full_file << jams::fmt::decimal << i * delta_t_ << " ";
    filter_full_file << jams::fmt::sci << full_memory_kernel[i] << "\n";
  }
  filter_full_file.close();

  std::ofstream filter_file(jams::output::full_path_filename("noise_filter_trunc.tsv"));
  filter_file << "delta_t_ps    filter_arb" << std::endl;
  for (auto i = 0; i < memory_kernel_.size(); ++i) {
    filter_file << jams::fmt::decimal << i * delta_t_ << " ";
    filter_file << jams::fmt::sci << memory_kernel_(i) << "\n";
  }
  filter_file.close();
}

void CpuGeneralFFTNoiseGenerator::output_thermostat_properties(std::ostream& os) const {
  if (lorentzian_gamma_ != 0.0) {
    os << "    lorentzian gamma (THz) " << std::fixed << lorentzian_gamma_ / kTwoPi << "\n";
  }
  if (lorentzian_omega0_ != 0.0) {
    os << "    lorentzian omega0 (THz) " << std::fixed << lorentzian_omega0_ / kTwoPi << "\n";
  }
  if (lorentzian_A_ != 0.0) {
    os << "    lorentzian A " << std::fixed << lorentzian_A_ << "\n";
  }

  os << "    max_omega (THz) " << std::fixed << max_omega_ / kTwoPi << "\n";
  os << "    delta_omega (THz) " << std::fixed << delta_omega_ / kTwoPi << "\n";
  os << "    delta_t " << std::scientific << delta_t_ << "\n";
  os << "    num_freq " << num_freq_ << "\n";
  os << "    num_trunc " << num_trunc_ << "\n";
}

std::vector<double> CpuGeneralFFTNoiseGenerator::discrete_real_fourier_transform(
    std::vector<double> x) {
  const int size = static_cast<int>(x.size());
  fftw_plan plan = fftw_plan_r2r_1d(
      size,
      x.data(),
      x.data(),
      FFTW_REDFT10,
      FFTW_ESTIMATE);

  fftw_execute(plan);
  fftw_destroy_plan(plan);

  for (double& value : x) {
    value /= 2 * size;
  }

  return x;
}

void CpuGeneralFFTNoiseGenerator::init_lorentzian(
    const CpuGeneralFFTNoiseGeneratorConfig& config) {
  if (std::isnan(config.lorentzian_gamma) || std::isnan(config.lorentzian_omega0)) {
    throw std::runtime_error(
        "lorentzian spectra require thermostat.lorentzian_gamma and thermostat.lorentzian_omega0");
  }

  lorentzian_gamma_ = kTwoPi * config.lorentzian_gamma;
  lorentzian_omega0_ = kTwoPi * config.lorentzian_omega0;
  lorentzian_A_ = pow4(lorentzian_omega0_) / lorentzian_gamma_;
}

double CpuGeneralFFTNoiseGenerator::classical_ohmic_spectrum(
    const double omega,
    const double temperature) {
  return 2.0 * kBoltzmannIU * temperature;
}

double CpuGeneralFFTNoiseGenerator::quantum_ohmic_spectrum(
    const double omega,
    const double temperature) {
  if (omega == 0.0) {
    return 2.0 * kBoltzmannIU * temperature;
  }

  const double x = (kHBarIU * std::abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return 2.0 * kBoltzmannIU * temperature * x * jams::maths::coth(x);
}

double CpuGeneralFFTNoiseGenerator::quantum_no_zero_ohmic_spectrum(
    const double omega,
    const double temperature) {
  if (omega == 0.0) {
    return 2.0 * kBoltzmannIU * temperature;
  }

  const double x = (kHBarIU * std::abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return 2.0 * kBoltzmannIU * temperature * x * (jams::maths::coth(x) - 1.0);
}

double CpuGeneralFFTNoiseGenerator::classical_lorentzian_spectrum(
    const double omega,
    const double temperature,
    const double omega0,
    const double gamma,
    const double A) {
  if (omega == 0.0) {
    return (2.0 * kBoltzmannIU * temperature) * (A * gamma) / pow4(omega0);
  }

  const double x = (kHBarIU * std::abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return kHBarIU * jams::maths::lorentzian(std::abs(omega), omega0, gamma, A) / x;
}

double CpuGeneralFFTNoiseGenerator::quantum_lorentzian_spectrum(
    const double omega,
    const double temperature,
    const double omega0,
    const double gamma,
    const double A) {
  if (omega == 0.0) {
    return (2.0 * kBoltzmannIU * temperature) * (A * gamma) / pow4(omega0);
  }

  const double x = (kHBarIU * std::abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return kHBarIU * jams::maths::lorentzian(std::abs(omega), omega0, gamma, A)
         * jams::maths::coth(x);
}

double CpuGeneralFFTNoiseGenerator::quantum_no_zero_lorentzian_spectrum(
    const double omega,
    const double temperature,
    const double omega0,
    const double gamma,
    const double A) {
  if (omega == 0.0) {
    return (2.0 * kBoltzmannIU * temperature) * (A * gamma) / pow4(omega0);
  }

  const double x = (kHBarIU * std::abs(omega)) / (2.0 * kBoltzmannIU * temperature);
  return kHBarIU * jams::maths::lorentzian(std::abs(omega), omega0, gamma, A)
         * (jams::maths::coth(x) - 1.0);
}
