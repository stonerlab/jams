#ifndef JAMS_TEST_QUANTUM_SPDE_NOISE_H
#define JAMS_TEST_QUANTUM_SPDE_NOISE_H

#include <cmath>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include <libconfig.h++>

#include "jams/common.h"
#include "jams/containers/multiarray.h"
#include "jams/core/globals.h"
#include "jams/core/thermostat.h"
#include "jams/helpers/consts.h"
#include "jams/thermostats/quantum_spde_noise.h"
#include "jams/thermostats/thermostat_quantum_spde.h"

namespace {

struct CpuQuantumSpdePairMoments {
  double mean0 = 0.0;
  double mean1 = 0.0;
  double var0 = 0.0;
  double cov01 = 0.0;
  double var1 = 0.0;
};

struct CpuQuantumSpdeTargetCovariance {
  double p00 = 0.0;
  double p01 = 0.0;
  double p11 = 0.0;
};

CpuQuantumSpdeTargetCovariance cpu_quantum_spde_covariance_from_cholesky(
    const jams::QuantumSpdeBoseCholesky& factor) {
  return {
      factor.l00 * factor.l00,
      factor.l00 * factor.l10,
      factor.l10 * factor.l10 + factor.l11 * factor.l11,
  };
}

CpuQuantumSpdeTargetCovariance cpu_quantum_spde_stationary_covariance(
    const double gamma, const double omega, const double h) {
  return cpu_quantum_spde_covariance_from_cholesky(
      jams::quantum_spde_stationary_bose_cholesky(gamma, omega, h));
}

CpuQuantumSpdePairMoments cpu_quantum_spde_pair_moments(
    const double* x0, const double* x1, const int count) {
  CpuQuantumSpdePairMoments moments;
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

void cpu_quantum_spde_expect_moments_near_target(
    const CpuQuantumSpdePairMoments& moments,
    const CpuQuantumSpdeTargetCovariance& target) {
  EXPECT_NEAR(moments.mean0, 0.0, 1.5e-2);
  EXPECT_NEAR(moments.mean1, 0.0, 1.5e-2);
  EXPECT_NEAR(moments.var0, target.p00, 3.0e-2);
  EXPECT_NEAR(moments.cov01, target.p01, 2.0e-2);
  EXPECT_NEAR(moments.var1, target.p11, 3.0e-2);
}

void cpu_quantum_spde_fill_sigma(jams::MultiArray<jams::Real, 1>& sigma) {
  auto sigma_view = sigma.mutable_host_view();
  auto* sigma_values = sigma_view.data();
  for (auto i = 0; i < sigma_view.size(); ++i) {
    sigma_values[i] = jams::Real{1.0};
  }
}

}  // namespace

class QuantumSpdeThermostatFactoryCpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    globals::num_spins = 2;
    globals::num_spins3 = 6;
    globals::alpha.resize(globals::num_spins).fill(jams::Real{0.1});
    globals::mus.resize(globals::num_spins).fill(jams::Real{1.0});
    globals::gyro.resize(globals::num_spins).fill(jams::Real{1.0});
    globals::sync_magnetic_moment_data();
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(R"(
      physics = {
        temperature = 0.0;
      };
      thermostat = {
        initialization = "zero";
        zero_point = false;
      };
    )");
  }

  void TearDown() override {
    globals::alpha.clear();
    globals::mus.clear();
    globals::inv_mus.clear();
    globals::num_magnetic_spins = 0;
    globals::gyro.clear();
    globals::num_spins = 0;
    globals::num_spins3 = 0;
    globals::config = nullptr;
  }
};

TEST(QuantumSpdeSharedNoiseTest, StationaryCovarianceSolvesDiscreteLyapunov) {
  const double h_values[] = {1.0e-4, 1.0e-2, 4.0e-2, 2.5e-1};

  for (const auto [gamma, omega] :
       {std::pair{jams::kQuantumSpdeGamma5, jams::kQuantumSpdeOmega5},
        std::pair{jams::kQuantumSpdeGamma6, jams::kQuantumSpdeOmega6}}) {
    for (const double h : h_values) {
      const auto target = cpu_quantum_spde_stationary_covariance(gamma, omega, h);

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

TEST(QuantumSpdeSharedNoiseTest, PrecomputedBoseUpdateMatchesExactHostUpdate) {
  constexpr double kEtaSample = 0.37;
  const double h_values[] = {1.0e-4, 1.0e-2, 4.0e-2, 2.5e-1};

  for (const auto [gamma, omega] :
       {std::pair{jams::kQuantumSpdeGamma5, jams::kQuantumSpdeOmega5},
        std::pair{6.0, 1.0},
        std::pair{4.0, 2.0}}) {
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

TEST(QuantumSpdeNoiseGeneratorCpuTest, StationaryInitializationSamplesTargetMoments) {
  jams::Jams::set_mode(jams::Mode::CPU);

  constexpr int kProcessCount = 1 << 16;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  jams::QuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false);

  generator.initialize_stationary(kTemperature);

  const double h = kDeltaTau * kTemperature;
  cpu_quantum_spde_expect_moments_near_target(
      cpu_quantum_spde_pair_moments(generator.zeta5().host_data(),
                                    generator.zeta5p().host_data(),
                                    kProcessCount),
      cpu_quantum_spde_stationary_covariance(
          jams::kQuantumSpdeGamma5, jams::kQuantumSpdeOmega5, h));
  cpu_quantum_spde_expect_moments_near_target(
      cpu_quantum_spde_pair_moments(generator.zeta6().host_data(),
                                    generator.zeta6p().host_data(),
                                    kProcessCount),
      cpu_quantum_spde_stationary_covariance(
          jams::kQuantumSpdeGamma6, jams::kQuantumSpdeOmega6, h));
}

TEST(QuantumSpdeNoiseGeneratorCpuTest, StationaryStateSurvivesBackToBackUpdates) {
  jams::Jams::set_mode(jams::Mode::CPU);

  constexpr int kProcessCount = 1 << 16;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kTemperature = 300.0;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  jams::QuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  cpu_quantum_spde_fill_sigma(sigma);

  generator.initialize_stationary(kTemperature);
  auto noise_view = noise.mutable_host_view();
  const auto sigma_view = sigma.host_view();
  for (auto i = 0; i < 64; ++i) {
    generator.update(noise_view.data(), sigma_view.data(), kTemperature);
  }

  const double h = kDeltaTau * kTemperature;
  cpu_quantum_spde_expect_moments_near_target(
      cpu_quantum_spde_pair_moments(generator.zeta5().host_data(),
                                    generator.zeta5p().host_data(),
                                    kProcessCount),
      cpu_quantum_spde_stationary_covariance(
          jams::kQuantumSpdeGamma5, jams::kQuantumSpdeOmega5, h));
  cpu_quantum_spde_expect_moments_near_target(
      cpu_quantum_spde_pair_moments(generator.zeta6().host_data(),
                                    generator.zeta6p().host_data(),
                                    kProcessCount),
      cpu_quantum_spde_stationary_covariance(
          jams::kQuantumSpdeGamma6, jams::kQuantumSpdeOmega6, h));
}

TEST(QuantumSpdeNoiseGeneratorCpuTest, ZeroTemperatureWithoutZeroPointClearsNoise) {
  jams::Jams::set_mode(jams::Mode::CPU);

  constexpr int kProcessCount = 4096;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  jams::QuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, false);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  cpu_quantum_spde_fill_sigma(sigma);
  auto noise_view = noise.mutable_host_view();
  for (auto i = 0; i < kProcessCount; ++i) {
    noise_view(i) = jams::Real{123.0};
  }

  const auto sigma_view = sigma.host_view();
  generator.update(noise_view.data(), sigma_view.data(), jams::Real{0.0});

  const auto noise_host = noise.host_view();
  for (auto i = 0; i < kProcessCount; ++i) {
    EXPECT_EQ(noise_host(i), jams::Real{0.0});
  }
}

TEST(QuantumSpdeNoiseGeneratorCpuTest, ZeroTemperatureWithZeroPointProducesNoise) {
  jams::Jams::set_mode(jams::Mode::CPU);

  constexpr int kProcessCount = 4096;
  constexpr double kTimestepPs = 1.0e-3;
  constexpr double kDeltaTau = (kTimestepPs * kBoltzmannIU) / kHBarIU;
  jams::QuantumSpdeNoiseGenerator generator(
      kProcessCount, kDeltaTau, 25.0 * kTwoPi, true);
  jams::MultiArray<jams::Real, 1> sigma(kProcessCount);
  jams::MultiArray<jams::Real, 1> noise(kProcessCount);
  cpu_quantum_spde_fill_sigma(sigma);

  auto noise_view = noise.mutable_host_view();
  const auto sigma_view = sigma.host_view();
  generator.initialize_stationary(jams::Real{0.0});
  generator.update(noise_view.data(), sigma_view.data(), jams::Real{0.0});

  double variance = 0.0;
  const auto noise_host = noise.host_view();
  for (auto i = 0; i < kProcessCount; ++i) {
    variance += static_cast<double>(noise_host(i)) * static_cast<double>(noise_host(i));
  }
  variance /= kProcessCount;
  EXPECT_GT(variance, 1.0e-12);
}

TEST_F(QuantumSpdeThermostatFactoryCpuTest, FactoryConstructsQuantumSpdeCpuThermostat) {
  std::unique_ptr<Thermostat> thermostat(
      Thermostat::create("quantum-spde-cpu", jams::Real{1.0e-3}));
  ASSERT_NE(thermostat, nullptr);

  thermostat->update();
  const auto* noise = thermostat->data();
  for (auto i = 0; i < globals::num_spins3; ++i) {
    EXPECT_EQ(noise[i], jams::Real{0.0});
  }
}

#endif  // JAMS_TEST_QUANTUM_SPDE_NOISE_H
