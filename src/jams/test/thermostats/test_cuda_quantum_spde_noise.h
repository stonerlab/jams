#ifndef JAMS_TEST_CUDA_QUANTUM_SPDE_NOISE_H
#define JAMS_TEST_CUDA_QUANTUM_SPDE_NOISE_H

#if HAS_CUDA

#include <cmath>
#include <utility>

#include <cuda_runtime_api.h>
#include <curand.h>
#include <gtest/gtest.h>

#include "jams/common.h"
#include "jams/containers/multiarray.h"
#include "jams/helpers/consts.h"
#include "jams/thermostats/cuda_quantum_spde_noise.h"

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

}  // namespace

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

#endif  // HAS_CUDA
#endif  // JAMS_TEST_CUDA_QUANTUM_SPDE_NOISE_H
