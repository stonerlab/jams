#ifndef JAMS_TEST_CUDA_QUANTUM_SPDE_NOISE_H
#define JAMS_TEST_CUDA_QUANTUM_SPDE_NOISE_H

#if HAS_CUDA

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <curand.h>
#include <gtest/gtest.h>
#include <libconfig.h++>

#include "jams/common.h"
#include "jams/containers/multiarray.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/thermostats/cuda_quantum_spde_noise.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde.h"

namespace {

struct PairMoments {
  double mean0 = 0.0;
  double mean1 = 0.0;
  double var0 = 0.0;
  double cov01 = 0.0;
  double var1 = 0.0;
};

struct TargetCovariance {
  double p00 = 0.0;
  double p01 = 0.0;
  double p11 = 0.0;
};

bool cuda_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void initialize_cuda_for_quantum_spde_tests(const unsigned long long seed) {
  jams::Jams::set_mode(jams::Mode::GPU);
  ASSERT_EQ(curandSetStream(jams::instance().curand_generator(), nullptr),
            CURAND_STATUS_SUCCESS);
  ASSERT_EQ(curandSetPseudoRandomGeneratorSeed(jams::instance().curand_generator(), seed),
            CURAND_STATUS_SUCCESS);
  ASSERT_EQ(curandSetGeneratorOffset(jams::instance().curand_generator(), 0),
            CURAND_STATUS_SUCCESS);
  ASSERT_EQ(curandGenerateSeeds(jams::instance().curand_generator()), CURAND_STATUS_SUCCESS);
}

TargetCovariance covariance_from_cholesky(const jams::QuantumSpdeBoseCholesky& factor) {
  return {
      factor.l00 * factor.l00,
      factor.l00 * factor.l10,
      factor.l10 * factor.l10 + factor.l11 * factor.l11,
  };
}

TargetCovariance stationary_covariance(const double gamma, const double omega,
                                       const double h) {
  return covariance_from_cholesky(
      jams::quantum_spde_stationary_bose_cholesky(gamma, omega, h));
}

PairMoments pair_moments(const double* x0, const double* x1, const int count) {
  PairMoments moments;
  for (auto i = 0; i < count; ++i) {
    moments.mean0 += x0[i];
    moments.mean1 += x1[i];
  }
  moments.mean0 /= count;
  moments.mean1 /= count;

  for (auto i = 0; i < count; ++i) {
    const double d0 = x0[i] - moments.mean0;
    const double d1 = x1[i] - moments.mean1;
    moments.var0 += d0 * d0;
    moments.cov01 += d0 * d1;
    moments.var1 += d1 * d1;
  }
  moments.var0 /= count;
  moments.cov01 /= count;
  moments.var1 /= count;
  return moments;
}

void expect_moments_near_target(const PairMoments& moments,
                                const TargetCovariance& target) {
  EXPECT_NEAR(moments.mean0, 0.0, 1.5e-2);
  EXPECT_NEAR(moments.mean1, 0.0, 1.5e-2);
  EXPECT_NEAR(moments.var0, target.p00, 3.0e-2);
  EXPECT_NEAR(moments.cov01, target.p01, 2.0e-2);
  EXPECT_NEAR(moments.var1, target.p11, 3.0e-2);
}

void fill_sigma(jams::MultiArray<jams::Real, 1>& sigma) {
  for (auto i = 0; i < sigma.size(); ++i) {
    sigma(i) = jams::Real{1.0};
  }
}

void generate_global_normal(jams::MultiArray<jams::Real, 1>& data) {
#ifdef DO_MIXED_PRECISION
  ASSERT_EQ(curandGenerateNormal(
      jams::instance().curand_generator(), data.mutable_device_data(), data.size(), 0.0, 1.0),
            CURAND_STATUS_SUCCESS);
#else
  ASSERT_EQ(curandGenerateNormalDouble(
      jams::instance().curand_generator(), data.mutable_device_data(), data.size(), 0.0, 1.0),
            CURAND_STATUS_SUCCESS);
#endif
}

void fill_deterministic_gaussian_pair(jams::MultiArray<jams::Real, 1>& eta5,
                                      jams::MultiArray<jams::Real, 1>& eta6) {
  constexpr double kInvUint32Range = 1.0 / 4294967296.0;
  std::uint32_t state = 0x9e3779b9U;
  auto next_uniform = [&state]() {
    state = 1664525U * state + 1013904223U;
    return (static_cast<double>(state) + 0.5) * kInvUint32Range;
  };

  for (std::size_t i = 0; i < eta5.size(); ++i) {
    const double radius = std::sqrt(-2.0 * std::log(next_uniform()));
    const double angle = kTwoPi * next_uniform();
    eta5(i) = static_cast<jams::Real>(radius * std::cos(angle));
    eta6(i) = static_cast<jams::Real>(radius * std::sin(angle));
  }
}

}  // namespace

class CudaQuantumSpdeThermostatCompletionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!cuda_device_available()) {
      return;
    }
    initialize_cuda_for_quantum_spde_tests(11235813ULL);
    globals::num_spins = kSpinCount;
    globals::num_spins3 = 3 * kSpinCount;
    globals::alpha.resize(kSpinCount).fill(jams::Real{0.1});
    globals::mus.resize(kSpinCount).fill(jams::Real{1.0});
    globals::gyro.resize(kSpinCount).fill(jams::Real{1.0});
    globals::sync_magnetic_moment_data();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(R"(
      physics = { temperature = 300.0; };
      thermostat = {
        initialization = "stationary";
        warmup = false;
        zero_point = false;
      };
    )");
  }

  void TearDown() override {
    globals::alpha.clear();
    globals::mus.clear();
    globals::inv_mus.clear();
    globals::gyro.clear();
    globals::num_magnetic_spins = 0;
    globals::num_spins = 0;
    globals::num_spins3 = 0;
    globals::config = nullptr;
  }

  static constexpr int kSpinCount = 8192;
};

TEST(QuantumSpdeNoiseGeneratorTest, StationaryCovarianceSolvesDiscreteLyapunov) {
  constexpr double kGamma5 = 5.0142;
  constexpr double kOmega5 = 2.7189;
  constexpr double kGamma6 = 3.2974;
  constexpr double kOmega6 = 1.2223;
  const double h_values[] = {1.0e-4, 1.0e-2, 4.0e-2, 2.5e-1};

  for (const auto [gamma, omega] :
       {std::pair{kGamma5, kOmega5}, std::pair{kGamma6, kOmega6}}) {
    for (const double h : h_values) {
      const auto target = stationary_covariance(gamma, omega, h);

      double z[2] = {1.0, 0.0};
      jams::quantum_spde_bose_exact_update_host(gamma, omega, 0.0, h, z);
      const double a00 = z[0];
      const double a10 = z[1];

      z[0] = 0.0;
      z[1] = 1.0;
      jams::quantum_spde_bose_exact_update_host(gamma, omega, 0.0, h, z);
      const double a01 = z[0];
      const double a11 = z[1];

      z[0] = 0.0;
      z[1] = 0.0;
      jams::quantum_spde_bose_exact_update_host(gamma, omega, 1.0, h, z);
      const double b0 = z[0];
      const double b1 = z[1];

      const double force_variance = 2.0 * gamma / h;
      const double q00 = force_variance * b0 * b0;
      const double q01 = force_variance * b0 * b1;
      const double q11 = force_variance * b1 * b1;

      const double ap00 = a00 * a00 * target.p00
          + 2.0 * a00 * a01 * target.p01
          + a01 * a01 * target.p11;
      const double ap01 = a00 * a10 * target.p00
          + (a00 * a11 + a01 * a10) * target.p01
          + a01 * a11 * target.p11;
      const double ap11 = a10 * a10 * target.p00
          + 2.0 * a10 * a11 * target.p01
          + a11 * a11 * target.p11;

      EXPECT_NEAR(target.p00, ap00 + q00, 2.0e-12);
      EXPECT_NEAR(target.p01, ap01 + q01, 2.0e-12);
      EXPECT_NEAR(target.p11, ap11 + q11, 2.0e-12);
    }
  }
}

TEST(QuantumSpdeNoiseGeneratorTest, PrecomputedBoseUpdateMatchesExactHostUpdate) {
  constexpr double kEtaSample = 0.37;
  const double h_values[] = {1.0e-4, 1.0e-2, 4.0e-2, 2.5e-1};

  for (const auto [gamma, omega] :
       {std::pair{5.0142, 2.7189}, std::pair{6.0, 1.0}, std::pair{4.0, 2.0}}) {
    for (const double h : h_values) {
      const auto coeffs = jams::quantum_spde_bose_update_coefficients(gamma, omega, h);
      const double eta0 = static_cast<double>(
          static_cast<jams::Real>(kEtaSample) * coeffs.eta_scale);
      const double force_eq = eta0 * coeffs.inv_omega2;

      double expected[2] = {0.125, -0.75};
      jams::quantum_spde_bose_exact_update_host(gamma, omega, eta0, h, expected);

      const double actual0 = coeffs.m00 * 0.125 + coeffs.m01 * -0.75
          + coeffs.force0 * force_eq;
      const double actual1 = coeffs.m10 * 0.125 + coeffs.m11 * -0.75
          + coeffs.force1 * force_eq;

      EXPECT_NEAR(expected[0], actual0, 2.0e-12);
      EXPECT_NEAR(expected[1], actual1, 2.0e-12);
    }
  }
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, CheckedSizingHelpersCoverLargeProcessCounts) {
  constexpr int kMaxInt = std::numeric_limits<int>::max();
  constexpr int kMaxSpinCount = kMaxInt / 3;

  EXPECT_EQ(jams::quantum_spde_cuda_process_count(kMaxSpinCount),
            kMaxSpinCount * 3);
  EXPECT_THROW({
    const auto process_count = jams::quantum_spde_cuda_process_count(kMaxSpinCount + 1);
    (void)process_count;
  }, std::overflow_error);
  EXPECT_THROW({
    const auto process_count = jams::quantum_spde_cuda_process_count(-1);
    (void)process_count;
  }, std::length_error);

  EXPECT_EQ(jams::quantum_spde_cuda_curand_buffer_count(0), std::size_t{0});
  EXPECT_EQ(jams::quantum_spde_cuda_curand_buffer_count(1), std::size_t{2});
  EXPECT_EQ(jams::quantum_spde_cuda_curand_buffer_count(2), std::size_t{2});
  EXPECT_EQ(jams::quantum_spde_cuda_curand_buffer_count(kMaxInt),
            static_cast<std::size_t>(kMaxInt) + 1);

  constexpr int kBlockSize = 128;
  EXPECT_EQ(jams::quantum_spde_cuda_grid_size(0, kBlockSize), 1);
  EXPECT_EQ(jams::quantum_spde_cuda_grid_size(kMaxInt, kBlockSize),
            static_cast<int>((static_cast<std::size_t>(kMaxInt)
                + kBlockSize - 1) / kBlockSize));
  EXPECT_THROW({
    const auto grid_size = jams::quantum_spde_cuda_grid_size(kMaxInt, 0);
    (void)grid_size;
  }, std::invalid_argument);
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, StationaryInitializationSamplesTargetMoments_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(1234567ULL);

  constexpr int kProcessCount = 1 << 16;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false, stream);

  generator.initialize_stationary(kTemperature);
  generator.synchronize();

  const double h = kDeltaTau * kTemperature;
  expect_moments_near_target(
      pair_moments(generator.zeta5().host_data(), generator.zeta5p().host_data(),
                   kProcessCount),
      stationary_covariance(5.0142, 2.7189, h));
  expect_moments_near_target(
      pair_moments(generator.zeta6().host_data(), generator.zeta6p().host_data(),
                   kProcessCount),
      stationary_covariance(3.2974, 1.2223, h));
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, StationaryStateSurvivesBackToBackUpdates_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(9876543ULL);

  constexpr int kProcessCount = 1 << 16;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  fill_sigma(sigma);

  generator.initialize_stationary(kTemperature);
  for (auto i = 0; i < 64; ++i) {
    generator.update(noise.mutable_device_data(), sigma.device_data(), kTemperature);
  }
  generator.synchronize();

  const double h = kDeltaTau * kTemperature;
  expect_moments_near_target(
      pair_moments(generator.zeta5().host_data(), generator.zeta5p().host_data(),
                   kProcessCount),
      stationary_covariance(5.0142, 2.7189, h));
  expect_moments_near_target(
      pair_moments(generator.zeta6().host_data(), generator.zeta6p().host_data(),
                   kProcessCount),
      stationary_covariance(3.2974, 1.2223, h));
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, ReplayedGaussianNoiseConvergesToTargetCovariance_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(16180339ULL);

  constexpr int kProcessCount = 1 << 15;
  constexpr int kStepCount = 256;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  jams::MultiArray<jams::Real, 1> eta5(kStepCount * kProcessCount);
  jams::MultiArray<jams::Real, 1> eta6(kStepCount * kProcessCount);
  fill_sigma(sigma);
  fill_deterministic_gaussian_pair(eta5, eta6);

  generator.initialize(jams::CudaQuantumSpdeNoiseGenerator::Initialization::Zero,
                       kTemperature);
  const auto* eta5_device = eta5.device_data();
  const auto* eta6_device = eta6.device_data();
  for (auto step = 0; step < kStepCount; ++step) {
    const auto offset = static_cast<std::size_t>(step) * kProcessCount;
    generator.update_with_gaussian_replay(
        noise.mutable_device_data(), sigma.device_data(), kTemperature,
        eta5_device + offset, eta6_device + offset);
  }
  generator.synchronize();

  const double h = kDeltaTau * kTemperature;
  expect_moments_near_target(
      pair_moments(generator.zeta5().host_data(), generator.zeta5p().host_data(),
                   kProcessCount),
      stationary_covariance(5.0142, 2.7189, h));
  expect_moments_near_target(
      pair_moments(generator.zeta6().host_data(), generator.zeta6p().host_data(),
                   kProcessCount),
      stationary_covariance(3.2974, 1.2223, h));
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, ExportTemporalSeriesWhenRequested_GPU) {
  const char* output_path = std::getenv("JAMS_SPDE_DIAGNOSTIC_CSV");
  if (output_path == nullptr || output_path[0] == '\0') {
    GTEST_SKIP() << "set JAMS_SPDE_DIAGNOSTIC_CSV to export a temporal series";
  }
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(27182818ULL);

  constexpr int kProcessCount = 16;
  constexpr int kStepCount = 1 << 16;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise_series(kStepCount * kProcessCount);
  fill_sigma(sigma);
  generator.initialize_stationary(kTemperature);

  auto* noise_device = noise_series.mutable_device_data();
  const auto* sigma_device = sigma.device_data();
  for (auto step = 0; step < kStepCount; ++step) {
    generator.update(noise_device + static_cast<std::size_t>(step) * kProcessCount,
                     sigma_device, kTemperature);
  }
  generator.synchronize();

  std::ofstream output(output_path);
  ASSERT_TRUE(output.good()) << "cannot open " << output_path;
  output << "time_ps";
  for (auto process = 0; process < kProcessCount; ++process) {
    output << ",noise_" << process;
  }
  output << '\n' << std::setprecision(17);
  const auto* samples = noise_series.host_data();
  for (auto step = 0; step < kStepCount; ++step) {
    output << step * kTimestepPs;
    for (auto process = 0; process < kProcessCount; ++process) {
      output << ',' << samples[static_cast<std::size_t>(step) * kProcessCount + process];
    }
    output << '\n';
  }
  ASSERT_TRUE(output.good()) << "failed while writing " << output_path;
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, ZeroInitializationConvergesTowardStationarity_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(24681357ULL);

  constexpr int kProcessCount = 1 << 15;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  fill_sigma(sigma);

  generator.initialize(jams::CudaQuantumSpdeNoiseGenerator::Initialization::Zero,
                       kTemperature);
  generator.warmup(256, kTemperature, noise.mutable_device_data(), sigma.device_data());
  generator.synchronize();

  const double h = kDeltaTau * kTemperature;
  expect_moments_near_target(
      pair_moments(generator.zeta5().host_data(), generator.zeta5p().host_data(),
                   kProcessCount),
      stationary_covariance(5.0142, 2.7189, h));
  expect_moments_near_target(
      pair_moments(generator.zeta6().host_data(), generator.zeta6p().host_data(),
                   kProcessCount),
      stationary_covariance(3.2974, 1.2223, h));
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, ZeroTemperatureWithoutZeroPointClearsNoise_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(13579ULL);

  constexpr int kProcessCount = 4096;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  fill_sigma(sigma);
  for (auto i = 0; i < kProcessCount; ++i) {
    noise(i) = jams::Real{123.0};
  }

  generator.update(noise.mutable_device_data(), sigma.device_data(), jams::Real{0.0});
  generator.synchronize();

  const auto* noise_host = noise.host_data();
  for (auto i = 0; i < kProcessCount; ++i) {
    EXPECT_EQ(noise_host[i], jams::Real{0.0});
  }
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, ZeroTemperatureWithZeroPointProducesNoise_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(97531ULL);

  constexpr int kProcessCount = 4096;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, true, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  fill_sigma(sigma);

  generator.initialize_stationary(jams::Real{0.0});
  generator.update(noise.mutable_device_data(), sigma.device_data(), jams::Real{0.0});
  generator.synchronize();

  double variance = 0.0;
  const auto* noise_host = noise.host_data();
  for (auto i = 0; i < kProcessCount; ++i) {
    variance += static_cast<double>(noise_host[i]) * static_cast<double>(noise_host[i]);
  }
  variance /= kProcessCount;
  EXPECT_GT(variance, 1.0e-12);
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, OddProcessCountWithZeroPointUsesPaddedCurandBuffers_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }
  initialize_cuda_for_quantum_spde_tests(314159ULL);

  constexpr int kProcessCount = 4097;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, true, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  fill_sigma(sigma);

  generator.initialize_stationary(jams::Real{0.0});
  generator.update(noise.mutable_device_data(), sigma.device_data(), jams::Real{0.0});
  generator.synchronize();

  for (auto component = 0; component < 4; ++component) {
    EXPECT_EQ(generator.zeta0_component(component).size(),
              static_cast<std::size_t>(kProcessCount));
  }

  double variance = 0.0;
  const auto* noise_host = noise.host_data();
  for (auto i = 0; i < kProcessCount; ++i) {
    variance += static_cast<double>(noise_host[i]) * static_cast<double>(noise_host[i]);
  }
  variance /= kProcessCount;
  EXPECT_GT(variance, 1.0e-12);
}

TEST(CudaQuantumSpdeNoiseGeneratorTest, InternalGeneratorDoesNotAdvanceGlobalCurand_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  constexpr unsigned long long kSeed = 86420ULL;
  constexpr int kGlobalSampleCount = 2048;
  constexpr int kProcessCount = 4096;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;

  initialize_cuda_for_quantum_spde_tests(kSeed);
  jams::MultiArray<jams::Real, 1> reference(kGlobalSampleCount);
  generate_global_normal(reference);
  const auto* reference_host = reference.host_data();

  initialize_cuda_for_quantum_spde_tests(kSeed);
  CudaStream stream(CudaStream::Priority::LOW);
  jams::CudaQuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false, stream);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  fill_sigma(sigma);
  generator.update(noise.mutable_device_data(), sigma.device_data(), kTemperature);
  generator.synchronize();

  jams::MultiArray<jams::Real, 1> actual(kGlobalSampleCount);
  generate_global_normal(actual);
  const auto* actual_host = actual.host_data();

  for (auto i = 0; i < kGlobalSampleCount; ++i) {
    EXPECT_EQ(actual_host[i], reference_host[i]);
  }
}

TEST_F(CudaQuantumSpdeThermostatCompletionTest,
       CompletionEventOrdersExternalNoiseConsumer_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  constexpr double kTimestepPs = 1.0e-3;
  CudaThermostatQuantumSpde thermostat(
      jams::Real{300.0}, jams::Real{0.0}, jams::Real{kTimestepPs}, kSpinCount);
  thermostat.update();
  thermostat.record_done();

  CudaStream consumer(CudaStream::Priority::LOW);
  thermostat.wait_on(consumer.get());
  std::vector<jams::Real> noise(3 * kSpinCount);
  ASSERT_EQ(cudaMemcpyAsync(
                noise.data(), thermostat.device_data(),
                noise.size() * sizeof(jams::Real), cudaMemcpyDeviceToHost,
                consumer.get()),
            cudaSuccess);
  consumer.synchronize();

  double sum = 0.0;
  double sum_squares = 0.0;
  for (const auto value : noise) {
    ASSERT_TRUE(std::isfinite(static_cast<double>(value)));
    sum += static_cast<double>(value);
    sum_squares += static_cast<double>(value) * static_cast<double>(value);
  }
  const double mean = sum / noise.size();
  const double variance = sum_squares / noise.size() - mean * mean;
  EXPECT_GT(variance, 0.0);
}

TEST_F(CudaQuantumSpdeThermostatCompletionTest,
       ConsumerCompletionEventOrdersNextThermostatWork_GPU) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  CudaThermostatQuantumSpde thermostat(
      jams::Real{300.0}, jams::Real{0.0}, jams::Real{1.0e-3}, kSpinCount);
  thermostat.update();
  thermostat.record_done();

  CudaStream consumer(CudaStream::Priority::LOW);
  thermostat.wait_on(consumer.get());
  std::vector<jams::Real> first_noise(3 * kSpinCount);
  ASSERT_EQ(cudaMemcpyAsync(
                first_noise.data(), thermostat.device_data(),
                first_noise.size() * sizeof(jams::Real), cudaMemcpyDeviceToHost,
                consumer.get()),
            cudaSuccess);

  std::atomic<bool> consumer_finished{false};
  ASSERT_EQ(cudaLaunchHostFunc(
                consumer.get(),
                [](void* data) {
                  std::this_thread::sleep_for(std::chrono::milliseconds(100));
                  static_cast<std::atomic<bool>*>(data)->store(
                      true, std::memory_order_release);
                },
                &consumer_finished),
            cudaSuccess);
  thermostat.record_consumed(consumer.get());

  // This immediate second update must wait for the delayed external consumer
  // to return ownership of the reusable noise buffer.
  thermostat.wait_for_consumed();
  thermostat.update();
  thermostat.record_done();
  thermostat.synchronize_done();
  EXPECT_TRUE(consumer_finished.load(std::memory_order_acquire));
}

#endif  // HAS_CUDA
#endif  // JAMS_TEST_CUDA_QUANTUM_SPDE_NOISE_H
