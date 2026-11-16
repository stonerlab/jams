#ifndef JAMS_TEST_GILBERT_PREFACTOR_H
#define JAMS_TEST_GILBERT_PREFACTOR_H

#include <array>
#include <cmath>
#include <memory>
#include <type_traits>

#include <gtest/gtest.h>
#include <libconfig.h++>

#include "jams/core/globals.h"
#include "jams/core/jams++.h"
#include "jams/core/solver.h"
#include "jams/core/thermostat.h"

class GilbertPrefactorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    globals::num_spins = 1;
    globals::num_spins3 = 3;
    globals::alpha.resize(1).fill(jams::Real{1.0});
    globals::mus.resize(1).fill(jams::Real{1.0});
    globals::inv_mus.resize(1).fill(jams::Real{1.0});
    globals::gyro.resize(1).fill(jams::Real{2.0});
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(R"(
      physics = {temperature = 100.0;};
      solver = {module = "llg-heun-cpu"; thermostat = "langevin-white-cpu";
                t_step = 1.0e-15; t_max = 1.0e-15; gilbert_prefactor = false;};
    )");
  }

  void TearDown() override {
    globals::alpha.clear();
    globals::mus.clear();
    globals::inv_mus.clear();
    globals::gyro.clear();
    globals::num_spins = 0;
    globals::num_spins3 = 0;
    globals::config.reset();
  }

  std::array<double, 3> white_field(bool gilbert_prefactor) {
    globals::config->lookup("solver.gilbert_prefactor") = gilbert_prefactor;
    const auto previous_rng = jams::instance().random_generator();
    jams::instance().random_generator().seed(2026u);
    std::unique_ptr<Solver> solver(Solver::create(globals::config->lookup("solver")));
    jams::instance().random_generator() = previous_rng;
    auto* thermostat = solver->thermostat();
    thermostat->update();
    return {thermostat->field(0, 0), thermostat->field(0, 1), thermostat->field(0, 2)};
  }
};

TEST_F(GilbertPrefactorTest, ClassicalLlgBathMatchesBothTorquePrefactors) {
  const auto gilbert = white_field(true);
  const auto bare = white_field(false);
  constexpr double tolerance = std::is_same_v<jams::Real, float> ? 2.0e-6 : 1.0e-13;
  double power = 0.0;
  for (int i = 0; i < 3; ++i) {
    power += gilbert[i] * gilbert[i];
    EXPECT_NEAR(bare[i], gilbert[i] / std::sqrt(2.0), tolerance * std::abs(gilbert[i]));
  }
  EXPECT_GT(power, 0.0);
  EXPECT_EQ(globals::gyro(0), 2.0);
  EXPECT_EQ(globals::alpha(0), 1.0);

  jams::MultiArray<jams::Real, 1> gyro_eff(1);
  gyro_eff(0) = 2.0;
  EXPECT_DOUBLE_EQ(jams::thermal_field_coupling(gyro_eff)[0], 0.25);
  EXPECT_DOUBLE_EQ(jams::thermal_field_coupling(gyro_eff, jams::ThermalTorque::additive)[0], 0.5);
  gyro_eff(0) = 1.0;
  EXPECT_DOUBLE_EQ(jams::thermal_field_coupling(gyro_eff)[0], 0.5);
  EXPECT_DOUBLE_EQ(jams::thermal_field_coupling(gyro_eff, jams::ThermalTorque::additive)[0], 1.0);
}

#endif
