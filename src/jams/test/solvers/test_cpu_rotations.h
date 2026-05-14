#ifndef JAMS_TEST_SOLVERS_TEST_CPU_ROTATIONS_H
#define JAMS_TEST_SOLVERS_TEST_CPU_ROTATIONS_H

#include <libconfig.h++>

#include "gtest/gtest.h"

#include "jams/core/globals.h"
#include "jams/helpers/exception.h"
#include "jams/solvers/cpu_rotations.h"

namespace {

constexpr double kAngleTolerance = 1e-12;

void reset_rotation_solver_globals() {
  globals::num_spins = 0;
  globals::num_spins3 = 0;
  globals::s.clear();
}

RotationSolver make_rotation_solver(const char* config_string, libconfig::Config& config) {
  config.readString(config_string);
  return RotationSolver(config.lookup("solver"));
}

}  // namespace

class RotationSolverAngleSpecTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reset_rotation_solver_globals();
  }
};

TEST_F(RotationSolverAngleSpecTest, LegacyCountsKeepFullInclusiveSweep) {
  libconfig::Config config;
  auto solver = make_rotation_solver(R"(
    solver = {
      module = "rotations-cpu";
      num_theta = 2;
      num_phi = 3;
    };
  )", config);

  EXPECT_EQ(solver.max_steps(), 6);

  auto coordinates = solver.monitor_coordinates();
  ASSERT_EQ(coordinates.size(), 3);
  EXPECT_EQ(coordinates[0], 0);
  EXPECT_NEAR(coordinates[1], 0.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 0.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_EQ(coordinates[0], 1);
  EXPECT_NEAR(coordinates[1], 180.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 0.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_EQ(coordinates[0], 2);
  EXPECT_NEAR(coordinates[1], 360.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 0.0, kAngleTolerance);
}

TEST_F(RotationSolverAngleSpecTest, ConstantThetaSweepsPhiRange) {
  libconfig::Config config;
  auto solver = make_rotation_solver(R"(
    solver = {
      module = "rotations-cpu";
      theta = { value_deg = 90.0; };
      phi = { start_deg = 0.0; stop_deg = 180.0; count = 3; };
    };
  )", config);

  EXPECT_EQ(solver.max_steps(), 3);

  auto coordinates = solver.monitor_coordinates();
  ASSERT_EQ(coordinates.size(), 3);
  EXPECT_NEAR(coordinates[1], 0.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 90.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_NEAR(coordinates[1], 90.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 90.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_NEAR(coordinates[1], 180.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 90.0, kAngleTolerance);
}

TEST_F(RotationSolverAngleSpecTest, ConstantPhiSweepsThetaRange) {
  libconfig::Config config;
  auto solver = make_rotation_solver(R"(
    solver = {
      module = "rotations-cpu";
      theta = { start_deg = 30.0; stop_deg = 90.0; count = 3; };
      phi = { value_deg = 45.0; };
    };
  )", config);

  EXPECT_EQ(solver.max_steps(), 3);

  auto coordinates = solver.monitor_coordinates();
  ASSERT_EQ(coordinates.size(), 3);
  EXPECT_NEAR(coordinates[1], 45.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 30.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_NEAR(coordinates[1], 45.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 60.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_NEAR(coordinates[1], 45.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 90.0, kAngleTolerance);
}

TEST_F(RotationSolverAngleSpecTest, ExplicitValuesAndOpenEndpointAreSupported) {
  libconfig::Config config;
  auto solver = make_rotation_solver(R"(
    solver = {
      module = "rotations-cpu";
      theta = { values_deg = [ 10.0, 20.0 ]; };
      phi = { start_deg = 0.0; stop_deg = 360.0; count = 4; endpoint = false; };
    };
  )", config);

  EXPECT_EQ(solver.max_steps(), 8);

  auto coordinates = solver.monitor_coordinates();
  ASSERT_EQ(coordinates.size(), 3);
  EXPECT_NEAR(coordinates[1], 0.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 10.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_NEAR(coordinates[1], 90.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 10.0, kAngleTolerance);
}

TEST_F(RotationSolverAngleSpecTest, RejectsAmbiguousAngleSpecification) {
  libconfig::Config config;
  config.readString(R"(
    solver = {
      module = "rotations-cpu";
      theta = { value_deg = 90.0; count = 3; };
    };
  )");

  EXPECT_THROW(RotationSolver(config.lookup("solver")), jams::ConfigException);
}

#endif  // JAMS_TEST_SOLVERS_TEST_CPU_ROTATIONS_H
