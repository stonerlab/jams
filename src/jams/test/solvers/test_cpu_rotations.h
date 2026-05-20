#ifndef JAMS_TEST_SOLVERS_TEST_CPU_ROTATIONS_H
#define JAMS_TEST_SOLVERS_TEST_CPU_ROTATIONS_H

#include <libconfig.h++>

#include "gtest/gtest.h"

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/utils.h"
#include "jams/solvers/cpu_rotations.h"
#include "jams/test/output.h"

namespace {

constexpr double kAngleTolerance = 1e-12;

void reset_rotation_solver_globals() {
  globals::num_spins = 0;
  globals::num_spins3 = 0;
  jams::util::force_deallocation(globals::s);
  jams::util::force_deallocation(globals::h);
  jams::util::force_deallocation(globals::ds_dt);
  jams::util::force_deallocation(globals::positions);
  jams::util::force_deallocation(globals::alpha);
  jams::util::force_deallocation(globals::mus);
  jams::util::force_deallocation(globals::gyro);
  globals::config = nullptr;
  if (globals::lattice) {
    delete globals::lattice;
    globals::lattice = nullptr;
  }
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

class RotationSolverTargetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reset_rotation_solver_globals();
    jams::testing::toggle_cout();
  }

  void TearDown() override {
    reset_rotation_solver_globals();
    jams::testing::toggle_cout();
  }

  void initialise_lattice(const char* config_string) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(config_string);
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
  }
};

TEST_F(RotationSolverTargetTest, MaterialAndPositionTargetsScanCartesianProduct) {
  initialise_lattice(R"(
    solver = {
      module = "rotations-cpu";
      rotation_targets = (
        {
          name = "A";
          position = 1;
          theta = { values_deg = [ 0.0, 90.0 ]; };
          phi = { value_deg = 0.0; };
        },
        {
          name = "B";
          materials = [ "B", "C" ];
          theta = { values_deg = [ 0.0, 180.0 ]; };
          phi = { value_deg = 90.0; };
        }
      );
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; },
      { name = "B"; moment = 1.0; spin = [0.0, 0.0, 1.0]; },
      { name = "C"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
    );

    unitcell = {
      symops = false;
      parameter = 1e-10;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0]),
        ("B", [0.5, 0.0, 0.0]),
        ("C", [0.0, 0.5, 0.0])
      );
    };

    lattice = {
      size = [1, 1, 1];
      periodic = [true, true, true];
    };
  )");

  RotationSolver solver(globals::config->lookup("solver"));
  EXPECT_EQ(solver.max_steps(), 4);

  const auto columns = solver.monitor_coordinate_columns();
  ASSERT_EQ(columns.size(), 5);
  EXPECT_EQ(columns[1].name, "theta_A_deg");
  EXPECT_EQ(columns[2].name, "phi_A_deg");
  EXPECT_EQ(columns[3].name, "theta_B_deg");
  EXPECT_EQ(columns[4].name, "phi_B_deg");

  ASSERT_TRUE(solver.is_running());
  auto coordinates = solver.monitor_coordinates();
  ASSERT_EQ(coordinates.size(), 5);
  EXPECT_NEAR(coordinates[1], 0.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[2], 0.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[3], 0.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[4], 90.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_NEAR(coordinates[1], 0.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[3], 180.0, kAngleTolerance);
  EXPECT_NEAR(globals::s(0, 2), 1.0, kAngleTolerance);
  EXPECT_NEAR(globals::s(1, 2), -1.0, kAngleTolerance);
  EXPECT_NEAR(globals::s(2, 2), -1.0, kAngleTolerance);

  solver.run();
  coordinates = solver.monitor_coordinates();
  EXPECT_NEAR(coordinates[1], 90.0, kAngleTolerance);
  EXPECT_NEAR(coordinates[3], 0.0, kAngleTolerance);
  EXPECT_NEAR(globals::s(0, 0), 1.0, kAngleTolerance);
  EXPECT_NEAR(globals::s(0, 2), 0.0, kAngleTolerance);
  EXPECT_NEAR(globals::s(1, 2), 1.0, kAngleTolerance);
  EXPECT_NEAR(globals::s(2, 2), 1.0, kAngleTolerance);
}

TEST_F(RotationSolverTargetTest, RejectsOverlappingTargets) {
  initialise_lattice(R"(
    solver = {
      module = "rotations-cpu";
      rotation_targets = (
        { name = "A"; position = 1; theta = 0.0; phi = 0.0; },
        { name = "B"; material = "A"; theta = 0.0; phi = 0.0; }
      );
    };

    materials = (
      { name = "A"; moment = 1.0; spin = [0.0, 0.0, 1.0]; }
    );

    unitcell = {
      symops = false;
      parameter = 1e-10;
      basis = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]);
      positions = (
        ("A", [0.0, 0.0, 0.0])
      );
    };

    lattice = {
      size = [1, 1, 1];
      periodic = [true, true, true];
    };
  )");

  EXPECT_THROW(RotationSolver(globals::config->lookup("solver")), jams::ConfigException);
}

#endif  // JAMS_TEST_SOLVERS_TEST_CPU_ROTATIONS_H
