#ifndef JAMS_TEST_SOLVERS_TEST_CPU_MONTE_CARLO_CONSTRAINED_H
#define JAMS_TEST_SOLVERS_TEST_CPU_MONTE_CARLO_CONSTRAINED_H

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include <libconfig.h++>

#include "gtest/gtest.h"

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/montecarlo.h"
#include "jams/helpers/utils.h"
#include "jams/solvers/cpu_monte_carlo_constrained.h"
#include "jams/test/output.h"

namespace {

constexpr double kConstrainedMCTestTolerance = 1.0e-8;

void reset_constrained_mc_globals() {
  globals::solver = nullptr;
  globals::num_spins = 0;
  globals::num_spins3 = 0;
  globals::num_magnetic_spins = 0;
  jams::util::force_deallocation(globals::s);
  jams::util::force_deallocation(globals::h);
  jams::util::force_deallocation(globals::ds_dt);
  jams::util::force_deallocation(globals::positions);
  jams::util::force_deallocation(globals::alpha);
  jams::util::force_deallocation(globals::mus);
  jams::util::force_deallocation(globals::inv_mus);
  jams::util::force_deallocation(globals::gyro);
  globals::config = nullptr;
  if (globals::lattice) {
    delete globals::lattice;
    globals::lattice = nullptr;
  }
}

std::string constrained_mc_config(
    const std::string& constraint_type,
    const double theta,
    const double phi,
    const bool auto_align) {
  std::ostringstream config;
  config << std::fixed << std::setprecision(15);
  config << R"(
    solver = {
      module = "monte-carlo-constrained-cpu";
      max_steps = 100;
      output_write_steps = 1;
      move_angle_sigma = 0.05;
      auto_align = )" << (auto_align ? "true" : "false") << ";\n";
  if (!constraint_type.empty()) {
    config << "      cmc_constraint_type = \"" << constraint_type << "\";\n";
  }
  config << "      cmc_constraint_theta = " << theta << ";\n"
         << "      cmc_constraint_phi = " << phi << ";\n"
         << R"(    };

    physics = {
      module = "empty";
      temperature = 1000000.0;
    };

    materials = (
      {
        name = "A";
        moment = 2.0;
        spin = [1.0, 0.0, 0.0];
        transform = ([1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]);
      },
      {
        name = "B";
        moment = 1.0;
        spin = [0.0, 1.0, 0.0];
        transform = ([0.0, -1.0, 0.0],
                     [1.0,  0.0, 0.0],
                     [0.0,  0.0, 1.0]);
      }
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
        ("B", [0.5, 0.0, 0.0])
      );
    };

    lattice = {
      size = [1, 1, 1];
      periodic = [true, true, true];
    };
  )";
  return config.str();
}

jams::Vec<double, 3> constrained_mc_total(const bool apply_material_transform) {
  jams::Vec<double, 3> total = {0.0, 0.0, 0.0};
  for (auto i = 0; i < globals::num_spins; ++i) {
    auto spin = jams::montecarlo::get_spin(i);
    if (apply_material_transform) {
      spin = globals::lattice->material(
          globals::lattice->lattice_site_material_id(i)).transform * spin;
    }
    total += globals::mus(i) * spin;
  }
  return total;
}

void expect_direction(
    const jams::Vec<double, 3>& vector,
    const jams::Vec<double, 3>& expected) {
  const auto direction = jams::normalize(vector);
  for (auto component = 0; component < 3; ++component) {
    EXPECT_NEAR(direction[component], expected[component], kConstrainedMCTestTolerance);
  }
}

void expect_unit_spins() {
  for (auto i = 0; i < globals::num_spins; ++i) {
    EXPECT_NEAR(jams::norm(jams::montecarlo::get_spin(i)), 1.0, kConstrainedMCTestTolerance);
  }
}

}  // namespace

class ConstrainedMCSolverConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reset_constrained_mc_globals();
    jams::testing::toggle_cout();
  }

  void TearDown() override {
    reset_constrained_mc_globals();
    jams::testing::toggle_cout();
  }

  std::unique_ptr<ConstrainedMCSolver> make_solver(
      const std::string& constraint_type,
      const double theta,
      const double phi,
      const bool auto_align) {
    globals::config = std::make_unique<libconfig::Config>();
    const auto config_text = constrained_mc_config(constraint_type, theta, phi, auto_align);
    globals::config->readString(config_text.c_str());
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    return std::make_unique<ConstrainedMCSolver>(globals::config->lookup("solver"));
  }

  void exercise_pair_moves(const std::string& constraint_type, const bool transformed) {
    constexpr double theta = 55.0;
    constexpr double phi = -35.0;
    auto solver = make_solver(constraint_type, theta, phi, true);
    solver->register_physics_module(Physics::create(globals::config->lookup("physics")));

    const auto initial_spin_0 = jams::montecarlo::get_spin(0);
    const auto initial_spin_1 = jams::montecarlo::get_spin(1);
    jams::instance().random_generator().seed(13579u);

    for (auto step = 0; step < 64; ++step) {
      ASSERT_NO_THROW(solver->run());
    }

    const auto expected = jams::spherical_to_cartesian_vector(
        1.0, deg_to_rad(theta), deg_to_rad(phi));
    expect_direction(constrained_mc_total(transformed), expected);
    expect_unit_spins();

    const auto final_spin_0 = jams::montecarlo::get_spin(0);
    const auto final_spin_1 = jams::montecarlo::get_spin(1);
    const double total_change = jams::norm(final_spin_0 - initial_spin_0)
        + jams::norm(final_spin_1 - initial_spin_1);
    EXPECT_GT(total_change, 1.0e-6);
  }
};

TEST_F(ConstrainedMCSolverConstraintTest, DefaultsToMaterialTransform) {
  EXPECT_NO_THROW(make_solver("", 90.0, 0.0, false));
  expect_direction(constrained_mc_total(true), {1.0, 0.0, 0.0});
}

TEST_F(ConstrainedMCSolverConstraintTest, MaterialTransformTypeIsCaseInsensitive) {
  EXPECT_NO_THROW(make_solver("MaTeRiAl_TrAnSfOrM", 90.0, 0.0, false));
}

TEST_F(ConstrainedMCSolverConstraintTest, MagnetisationUsesMomentWeightedUntransformedSpins) {
  const auto phi = rad_to_deg(std::atan2(1.0, 2.0));
  EXPECT_NO_THROW(make_solver("MaGnEtIsAtIoN", 90.0, phi, false));
  expect_direction(
      constrained_mc_total(false),
      {2.0 / std::sqrt(5.0), 1.0 / std::sqrt(5.0), 0.0});
}

TEST_F(ConstrainedMCSolverConstraintTest, ModesRejectTheOtherCollectiveVectorDirection) {
  const auto magnetisation_phi = rad_to_deg(std::atan2(1.0, 2.0));
  EXPECT_THROW(make_solver("magnetisation", 90.0, 0.0, false), std::runtime_error);

  reset_constrained_mc_globals();
  EXPECT_THROW(
      make_solver("material_transform", 90.0, magnetisation_phi, false),
      std::runtime_error);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsUnknownConstraintType) {
  EXPECT_THROW(make_solver("sublattice", 90.0, 0.0, false), jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, MaterialTransformAlignmentOccursInOrderParameterSpace) {
  auto solver = make_solver("material_transform", 0.0, 0.0, true);
  expect_direction(constrained_mc_total(true), {0.0, 0.0, 1.0});
  expect_unit_spins();
}

TEST_F(ConstrainedMCSolverConstraintTest, MagnetisationAlignmentIgnoresMaterialTransforms) {
  auto solver = make_solver("magnetisation", 0.0, 0.0, true);
  expect_direction(constrained_mc_total(false), {0.0, 0.0, 1.0});
  expect_unit_spins();
}

TEST_F(ConstrainedMCSolverConstraintTest, PairMovesPreserveMagnetisationConstraint) {
  exercise_pair_moves("magnetisation", false);
}

TEST_F(ConstrainedMCSolverConstraintTest, PairMovesPreserveMaterialTransformConstraint) {
  exercise_pair_moves("material_transform", true);
}

#endif  // JAMS_TEST_SOLVERS_TEST_CPU_MONTE_CARLO_CONSTRAINED_H
