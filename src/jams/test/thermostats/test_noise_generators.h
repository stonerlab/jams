#ifndef JAMS_TEST_THERMOSTATS_TEST_NOISE_GENERATORS_H
#define JAMS_TEST_THERMOSTATS_TEST_NOISE_GENERATORS_H

#include <gtest/gtest.h>

#if HAS_CUDA

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include <curand.h>

#include "jams/common.h"
#include "jams/core/noise_generator.h"
#include "jams/helpers/consts.h"
#include "jams/interface/fft.h"
#include "jams/thermostats/cuda_thermostat_general_fft.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

namespace {

constexpr double kNoiseTestTemperature = 300.0;
constexpr double kNoiseTestTimeStepPs = 1.0e-3;
constexpr int kNoiseSampleCount = 5000;
constexpr int kQuantumBurnInSteps = 0;
constexpr int kQuantumStride = 8;
constexpr int kGeneralFftBurnInSteps = 100;
constexpr int kGeneralFftStride = 4;
constexpr int kQuantumPsdBurnInSteps = 0;
constexpr int kQuantumPsdSegmentLength = 1024;
constexpr int kQuantumPsdSegments = 16;
constexpr double kQuantumPsdOmegaMin = 0.4;
constexpr double kQuantumPsdOmegaMax = 8.0;
constexpr int kQuantumPsdBinCount = 10;
constexpr int kQuantumInitialSampleVectors = 4096;

bool have_noise_generator_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void initialise_noise_generator_test_rng(const uint64_t seed) {
  jams::Jams::set_mode(jams::Mode::GPU);
  jams::instance().random_generator().seed(seed);

  ASSERT_EQ(
      curandSetPseudoRandomGeneratorSeed(jams::instance().curand_generator(), seed),
      CURAND_STATUS_SUCCESS);
}

struct SampleMoments {
  double mean = 0.0;
  double variance = 0.0;
  double frac_within_one_sigma = 0.0;
  double frac_within_two_sigma = 0.0;
};

struct PowerSpectrum {
  std::vector<double> omega;
  std::vector<double> density;
};

struct BinnedSpectrumComparison {
  double rms_relative_error = 0.0;
  double max_relative_error = 0.0;
  int populated_bins = 0;
};

SampleMoments compute_sample_moments(const std::vector<double>& samples,
                                     const double normalization_stddev = 0.0) {
  SampleMoments moments;
  for (const auto value : samples) {
    moments.mean += value;
  }
  moments.mean /= static_cast<double>(samples.size());

  for (const auto value : samples) {
    const double centered = value - moments.mean;
    moments.variance += centered * centered;
  }

  moments.variance /= static_cast<double>(samples.size() - 1);

  const double sigma = normalization_stddev > 0.0
      ? normalization_stddev
      : std::sqrt(moments.variance);
  for (const auto value : samples) {
    const double normalized = value / sigma;
    if (std::abs(normalized) <= 1.0) {
      moments.frac_within_one_sigma += 1.0;
    }
    if (std::abs(normalized) <= 2.0) {
      moments.frac_within_two_sigma += 1.0;
    }
  }

  moments.frac_within_one_sigma /= static_cast<double>(samples.size());
  moments.frac_within_two_sigma /= static_cast<double>(samples.size());
  return moments;
}

SampleMoments sample_noise_distribution(NoiseGenerator& generator,
                                        const int burn_in_steps,
                                        const int stride,
                                        const int sample_count,
                                        const double normalization_stddev = 0.0) {
  for (int step = 0; step < burn_in_steps; ++step) {
    generator.update();
  }
  generator.record_done();
  generator.synchronize_done();

  std::vector<double> samples;
  samples.reserve(sample_count);

  for (int sample = 0; sample < sample_count; ++sample) {
    for (int step = 0; step < stride; ++step) {
      generator.update();
    }
    generator.record_done();
    generator.synchronize_done();
    samples.push_back(generator.field(0, 0));
  }
  return compute_sample_moments(samples, normalization_stddev);
}

std::vector<double> sample_noise_time_series(NoiseGenerator& generator,
                                             const int burn_in_steps,
                                             const int sample_count,
                                             const double scale = 1.0) {
  for (int step = 0; step < burn_in_steps; ++step) {
    generator.update();
  }
  generator.record_done();
  generator.synchronize_done();

  std::vector<double> samples;
  samples.reserve(sample_count);
  for (int sample = 0; sample < sample_count; ++sample) {
    generator.update();
    generator.record_done();
    generator.synchronize_done();
    samples.push_back(scale * generator.field(0, 0));
  }
  return samples;
}

PowerSpectrum welch_power_spectrum(const std::vector<double>& samples,
                                   const double sample_step,
                                   const int segment_length) {
  const int num_segments = static_cast<int>(samples.size()) / segment_length;
  if (num_segments <= 0) {
    throw std::runtime_error("PSD estimate requires at least one complete segment");
  }

  std::vector<double> segment(segment_length, 0.0);
  std::vector<std::complex<double>> spectrum(segment_length / 2 + 1);
  auto* out = reinterpret_cast<fftw_complex*>(spectrum.data());
  fftw_plan plan = fftw_plan_dft_r2c_1d(segment_length, segment.data(), out, FFTW_ESTIMATE);

  PowerSpectrum result;
  result.omega.resize(segment_length / 2 + 1);
  result.density.assign(segment_length / 2 + 1, 0.0);

  const double normalization = sample_step / static_cast<double>(segment_length);
  const double omega_step = (2.0 * kPi) / (static_cast<double>(segment_length) * sample_step);

  for (int k = 0; k <= segment_length / 2; ++k) {
    result.omega[k] = omega_step * static_cast<double>(k);
  }

  for (int segment_index = 0; segment_index < num_segments; ++segment_index) {
    const auto begin = samples.begin() + segment_index * segment_length;
    const auto end = begin + segment_length;
    const double mean = std::accumulate(begin, end, 0.0) / static_cast<double>(segment_length);
    for (int i = 0; i < segment_length; ++i) {
      segment[i] = *(begin + i) - mean;
    }

    fftw_execute(plan);
    for (int k = 0; k <= segment_length / 2; ++k) {
      result.density[k] += normalization * std::norm(spectrum[k]);
    }
  }

  fftw_destroy_plan(plan);

  for (auto& value : result.density) {
    value /= static_cast<double>(num_segments);
  }

  return result;
}

double savin_no_zero_fit_psd(const double reduced_omega) {
  constexpr double kC5 = 1.8315;
  constexpr double kGamma5 = 5.0142;
  constexpr double kOmega5 = 2.7189;
  constexpr double kC6 = 0.3429;
  constexpr double kGamma6 = 3.2974;
  constexpr double kOmega6 = 1.2223;

  const auto fitted_component = [&](const double coefficient,
                                    const double gamma,
                                    const double omega0) {
    const double denominator =
        std::pow(omega0 * omega0 - reduced_omega * reduced_omega, 2)
        + reduced_omega * reduced_omega * gamma * gamma;
    return (2.0 * coefficient * coefficient * gamma) / denominator;
  };

  return fitted_component(kC5, kGamma5, kOmega5)
         + fitted_component(kC6, kGamma6, kOmega6);
}

BinnedSpectrumComparison compare_binned_relative_error(
    const PowerSpectrum& measured,
    const double omega_min,
    const double omega_max,
    const int num_bins,
    const std::function<double(double)>& analytic_density) {
  BinnedSpectrumComparison comparison;

  const double bin_width = (omega_max - omega_min) / static_cast<double>(num_bins);
  for (int bin = 0; bin < num_bins; ++bin) {
    const double bin_start = omega_min + static_cast<double>(bin) * bin_width;
    const double bin_end = bin_start + bin_width;

    double measured_sum = 0.0;
    double analytic_sum = 0.0;
    int count = 0;
    for (size_t i = 0; i < measured.omega.size(); ++i) {
      const double omega = measured.omega[i];
      if (omega < bin_start || omega >= bin_end) {
        continue;
      }
      measured_sum += measured.density[i];
      analytic_sum += analytic_density(omega);
      ++count;
    }

    if (count == 0 || analytic_sum <= 0.0) {
      continue;
    }

    const double measured_mean = measured_sum / static_cast<double>(count);
    const double analytic_mean = analytic_sum / static_cast<double>(count);
    const double relative_error = std::abs(measured_mean - analytic_mean) / analytic_mean;

    comparison.rms_relative_error += relative_error * relative_error;
    comparison.max_relative_error = std::max(comparison.max_relative_error, relative_error);
    ++comparison.populated_bins;
  }

  if (comparison.populated_bins > 0) {
    comparison.rms_relative_error =
        std::sqrt(comparison.rms_relative_error / static_cast<double>(comparison.populated_bins));
  }

  return comparison;
}

TEST(QuantumSpdeNoiseGeneratorTest, MarginalDistributionIsGaussian) {
  if (!have_noise_generator_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  initialise_noise_generator_test_rng(0x1234ULL);

  CudaQuantumSpdeNoiseGeneratorConfig config;
  config.zero_point = false;

  CudaQuantumSpdeNoiseGenerator generator(
      kNoiseTestTemperature, kNoiseTestTimeStepPs, 1, config);
  const double expected_variance = generator.stationary_variance();
  ASSERT_TRUE(std::isfinite(expected_variance));
  ASSERT_GT(expected_variance, 0.0);

  const auto moments = sample_noise_distribution(
      generator,
      kQuantumBurnInSteps,
      kQuantumStride,
      kNoiseSampleCount);

  EXPECT_NEAR(moments.mean, 0.0, 0.08 * std::sqrt(moments.variance));
  EXPECT_NEAR(moments.frac_within_one_sigma, 0.682689492137, 0.05);
  EXPECT_NEAR(moments.frac_within_two_sigma, 0.954499736104, 0.03);
}

TEST(QuantumSpdeNoiseGeneratorTest, FirstStepIsStationaryWithoutWarmup) {
  if (!have_noise_generator_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  initialise_noise_generator_test_rng(0x2345ULL);

  CudaQuantumSpdeNoiseGeneratorConfig config;
  config.zero_point = false;

  CudaQuantumSpdeNoiseGenerator generator(
      kNoiseTestTemperature, kNoiseTestTimeStepPs, kQuantumInitialSampleVectors, config);
  const double expected_variance = generator.stationary_variance();

  generator.update();
  generator.record_done();
  generator.synchronize_done();

  std::vector<double> samples;
  samples.reserve(kQuantumInitialSampleVectors);
  for (int i = 0; i < kQuantumInitialSampleVectors; ++i) {
    samples.push_back(generator.field(i, 0));
  }

  const auto moments = compute_sample_moments(samples, std::sqrt(expected_variance));
  EXPECT_NEAR(moments.mean, 0.0, 0.06 * std::sqrt(expected_variance));
  EXPECT_NEAR(moments.variance, expected_variance, 0.10 * expected_variance);
  EXPECT_NEAR(moments.frac_within_one_sigma, 0.682689492137, 0.04);
  EXPECT_NEAR(moments.frac_within_two_sigma, 0.954499736104, 0.025);
}

TEST(GeneralFftNoiseGeneratorTest, MarginalDistributionMatchesAnalyticGaussian) {
  if (!have_noise_generator_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  initialise_noise_generator_test_rng(0x5678ULL);

  CudaGeneralFFTNoiseGeneratorConfig config;
  config.spectrum = "classical";
  config.write_diagnostics = false;

  CudaGeneralFFTNoiseGenerator generator(
      kNoiseTestTemperature, kNoiseTestTimeStepPs, 1, config);
  const double expected_variance = generator.stationary_variance();
  ASSERT_GT(generator.memory_depth(), 0);
  ASSERT_TRUE(std::isfinite(expected_variance));
  ASSERT_GT(expected_variance, 0.0);

  const auto moments = sample_noise_distribution(
      generator,
      kGeneralFftBurnInSteps,
      kGeneralFftStride,
      kNoiseSampleCount,
      std::sqrt(expected_variance));

  EXPECT_NEAR(moments.mean, 0.0, 0.06 * std::sqrt(expected_variance));
  EXPECT_NEAR(moments.variance, expected_variance, 0.12 * expected_variance);
  EXPECT_NEAR(moments.frac_within_one_sigma, 0.682689492137, 0.06);
  EXPECT_NEAR(moments.frac_within_two_sigma, 0.954499736104, 0.04);
}

TEST(QuantumSpdeNoiseGeneratorTest, PowerSpectralDensityMatchesNoZeroTarget) {
  if (!have_noise_generator_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  initialise_noise_generator_test_rng(0x9abcULL);

  CudaQuantumSpdeNoiseGeneratorConfig config;
  config.zero_point = false;

  CudaQuantumSpdeNoiseGenerator generator(
      kNoiseTestTemperature, kNoiseTestTimeStepPs, 1, config);

  const auto reduced_timestep =
      kNoiseTestTimeStepPs * kBoltzmannIU * kNoiseTestTemperature / kHBarIU;
  const auto samples = sample_noise_time_series(
      generator,
      kQuantumPsdBurnInSteps,
      kQuantumPsdSegmentLength * kQuantumPsdSegments,
      1.0 / kNoiseTestTemperature);
  const auto psd = welch_power_spectrum(samples, reduced_timestep, kQuantumPsdSegmentLength);
  const auto comparison = compare_binned_relative_error(
      psd,
      kQuantumPsdOmegaMin,
      kQuantumPsdOmegaMax,
      kQuantumPsdBinCount,
      savin_no_zero_fit_psd);

  ASSERT_GE(comparison.populated_bins, kQuantumPsdBinCount - 1);
  EXPECT_LT(comparison.rms_relative_error, 0.18);
  EXPECT_LT(comparison.max_relative_error, 0.35);
}

}  // namespace

#endif  // HAS_CUDA

#endif  // JAMS_TEST_THERMOSTATS_TEST_NOISE_GENERATORS_H
