#ifndef JAMS_TEST_SOLVERS_TEST_CPU_MONTE_CARLO_CONSTRAINED_H
#define JAMS_TEST_SOLVERS_TEST_CPU_MONTE_CARLO_CONSTRAINED_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
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
    const bool auto_align,
    const std::string& constraint_mode = "",
    const std::optional<jams::Vec<double, 3>>& spiral_wavevector = std::nullopt,
    const std::optional<jams::Vec<double, 3>>& spiral_axis = std::nullopt,
    const jams::Vec<int, 3>& lattice_size = {1, 1, 1},
    const jams::Vec<bool, 3>& periodic = {true, true, true}) {
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
  if (!constraint_mode.empty()) {
    config << "      cmc_constraint_mode = \"" << constraint_mode << "\";\n";
  }
  if (spiral_wavevector.has_value()) {
    config << "      cmc_spiral_wavevector = ["
           << (*spiral_wavevector)[0] << ", "
           << (*spiral_wavevector)[1] << ", "
           << (*spiral_wavevector)[2] << "];\n";
  }
  if (spiral_axis.has_value()) {
    config << "      cmc_spiral_axis = ["
           << (*spiral_axis)[0] << ", "
           << (*spiral_axis)[1] << ", "
           << (*spiral_axis)[2] << "];\n";
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
      size = [)" << lattice_size[0] << ", " << lattice_size[1] << ", "
         << lattice_size[2] << "];\n"
         << "      periodic = [" << (periodic[0] ? "true" : "false") << ", "
         << (periodic[1] ? "true" : "false") << ", "
         << (periodic[2] ? "true" : "false") << "];\n"
         << R"(    };
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

jams::Vec<double, 3> constrained_mc_plane_total(
    const int propagation_direction,
    const int plane_coordinate,
    const bool apply_material_transform) {
  jams::Vec<double, 3> total = {0.0, 0.0, 0.0};
  for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    if (globals::lattice->cell_offset(spin_index)[propagation_direction]
        != plane_coordinate) {
      continue;
    }
    auto spin = jams::montecarlo::get_spin(spin_index);
    if (apply_material_transform) {
      spin = globals::lattice->material(
          globals::lattice->lattice_site_material_id(spin_index)).transform * spin;
    }
    total += globals::mus(spin_index) * spin;
  }
  return total;
}

jams::Vec<double, 3> constrained_mc_cell_total(
    const jams::Vec<int, 3>& cell_offset,
    const bool apply_material_transform) {
  jams::Vec<double, 3> total = {0.0, 0.0, 0.0};
  for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    if (globals::lattice->cell_offset(spin_index) != cell_offset) {
      continue;
    }
    auto spin = jams::montecarlo::get_spin(spin_index);
    if (apply_material_transform) {
      spin = globals::lattice->material(
          globals::lattice->lattice_site_material_id(spin_index)).transform * spin;
    }
    total += globals::mus(spin_index) * spin;
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

void add_vec_setting(
    libconfig::Setting& parent,
    const char* name,
    const jams::Vec<double, 3>& value) {
  auto& setting = parent.add(name, libconfig::Setting::TypeArray);
  for (auto component = 0; component < 3; ++component) {
    setting.add(libconfig::Setting::TypeFloat) = value[component];
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

  std::unique_ptr<ConstrainedMCSolver> make_solver_from_config(
      const std::string& config_text,
      const std::optional<double> constraint_tolerance = std::nullopt,
      const std::function<void()>& prepare_spins = {}) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(config_text.c_str());
    if (constraint_tolerance.has_value()) {
      globals::config->lookup("solver")
          .add("cmc_constraint_tolerance", libconfig::Setting::TypeFloat) = *constraint_tolerance;
    }
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    if (prepare_spins) {
      prepare_spins();
    }
    return std::make_unique<ConstrainedMCSolver>(globals::config->lookup("solver"));
  }

  std::unique_ptr<ConstrainedMCSolver> make_solver(
      const std::string& constraint_type,
      const double theta,
      const double phi,
      const bool auto_align,
      const std::optional<double> constraint_tolerance = std::nullopt,
      const std::function<void()>& prepare_spins = {}) {
    const auto config_text = constrained_mc_config(constraint_type, theta, phi, auto_align);
    return make_solver_from_config(config_text, constraint_tolerance, prepare_spins);
  }

  std::unique_ptr<ConstrainedMCSolver> make_spiral_solver(
      const std::string& constraint_type,
      const double theta,
      const double phi,
      const bool auto_align,
      const jams::Vec<double, 3>& wavevector,
      const jams::Vec<double, 3>& axis,
      const jams::Vec<int, 3>& lattice_size = {4, 2, 1},
      const jams::Vec<bool, 3>& periodic = {true, true, true},
      const std::optional<double> constraint_tolerance = std::nullopt,
      const std::function<void()>& prepare_spins = {}) {
    const auto config_text = constrained_mc_config(
        constraint_type,
        theta,
        phi,
        auto_align,
        "spin_spiral",
        wavevector,
        axis,
        lattice_size,
        periodic);
    return make_solver_from_config(config_text, constraint_tolerance, prepare_spins);
  }

  std::unique_ptr<ConstrainedMCSolver> make_programmatic_spiral_solver(
      const jams::Vec<double, 3>& wavevector,
      const jams::Vec<double, 3>& axis) {
    globals::config = std::make_unique<libconfig::Config>();
    const auto config_text = constrained_mc_config(
        "magnetisation", 90.0, 0.0, true, "", std::nullopt, std::nullopt,
        {4, 2, 1}, {true, true, true});
    globals::config->readString(config_text.c_str());
    auto& solver_settings = globals::config->lookup("solver");
    solver_settings.add("cmc_constraint_mode", libconfig::Setting::TypeString)
        = "spin_spiral";
    add_vec_setting(solver_settings, "cmc_spiral_wavevector", wavevector);
    add_vec_setting(solver_settings, "cmc_spiral_axis", axis);
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    return std::make_unique<ConstrainedMCSolver>(solver_settings);
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

  void exercise_spiral_pair_moves(
      const std::string& constraint_type,
      const bool transformed) {
    constexpr double theta = 90.0;
    constexpr double phi = 0.0;
    const jams::Vec<double, 3> wavevector = {0.25, 0.0, 0.0};
    const jams::Vec<double, 3> axis = {0.0, 0.0, 1.0};
    auto solver = make_spiral_solver(
        constraint_type, theta, phi, true, wavevector, axis);
    solver->register_physics_module(Physics::create(globals::config->lookup("physics")));

    jams::instance().random_generator().seed(24680u);
    for (auto step = 0; step < 128; ++step) {
      ASSERT_NO_THROW(solver->run());
    }

    const jams::Vec<double, 3> reference = {1.0, 0.0, 0.0};
    double maximum_cell_angular_error = 0.0;
    for (auto plane = 0; plane < 4; ++plane) {
      const auto expected = rotation_matrix_from_axis_angle(
          axis, kTwoPi * wavevector[0] * static_cast<double>(plane)) * reference;
      expect_direction(constrained_mc_plane_total(0, plane, transformed), expected);

      for (auto transverse_cell = 0; transverse_cell < 2; ++transverse_cell) {
        const auto cell_vector = constrained_mc_cell_total(
            {plane, transverse_cell, 0}, transformed);
        const double angular_error = std::atan2(
            jams::norm(jams::cross(cell_vector, expected)),
            jams::dot(cell_vector, expected));
        maximum_cell_angular_error = std::max(
            maximum_cell_angular_error, angular_error);
      }
    }
    expect_unit_spins();
    EXPECT_GT(maximum_cell_angular_error, 1.0e-6);
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

TEST_F(ConstrainedMCSolverConstraintTest, ConstraintModeIsCaseInsensitive) {
  const auto config_text = constrained_mc_config(
      "material_transform", 90.0, 0.0, false, "GlObAl");
  EXPECT_NO_THROW(make_solver_from_config(config_text));
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsUnknownConstraintMode) {
  const auto config_text = constrained_mc_config(
      "material_transform", 90.0, 0.0, false, "unit_cell");
  EXPECT_THROW(make_solver_from_config(config_text), jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralRequiresWavevectorAndAxis) {
  const jams::Vec<double, 3> wavevector = {0.25, 0.0, 0.0};
  const jams::Vec<double, 3> axis = {0.0, 0.0, 1.0};
  const auto missing_axis = constrained_mc_config(
      "magnetisation", 90.0, 0.0, true, "spin_spiral", wavevector,
      std::nullopt, {4, 2, 1});
  EXPECT_THROW(make_solver_from_config(missing_axis), jams::ConfigException);

  reset_constrained_mc_globals();
  const auto missing_wavevector = constrained_mc_config(
      "magnetisation", 90.0, 0.0, true, "spin_spiral", std::nullopt,
      axis, {4, 2, 1});
  EXPECT_THROW(make_solver_from_config(missing_wavevector), jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, GlobalModeRejectsSpiralSettings) {
  const auto config_text = constrained_mc_config(
      "magnetisation", 90.0, 0.0, true, "global",
      jams::Vec<double, 3>{0.25, 0.0, 0.0},
      jams::Vec<double, 3>{0.0, 0.0, 1.0}, {4, 2, 1});
  EXPECT_THROW(make_solver_from_config(config_text), jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsInvalidSpinSpiralVectors) {
  for (const auto wavevector : {
           jams::Vec<double, 3>{0.0, 0.0, 0.0},
           jams::Vec<double, 3>{0.25, 0.25, 0.0}}) {
    SCOPED_TRACE(wavevector);
    EXPECT_THROW(
        make_spiral_solver(
            "magnetisation", 90.0, 0.0, true, wavevector, {0.0, 0.0, 1.0}),
        jams::ConfigException);
    reset_constrained_mc_globals();
  }

  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, true, {0.25, 0.0, 0.0}, {0.0, 0.0, 0.0}),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsNonFiniteSpinSpiralVectors) {
  const double infinity = std::numeric_limits<double>::infinity();
  EXPECT_THROW(
      make_programmatic_spiral_solver({infinity, 0.0, 0.0}, {0.0, 0.0, 1.0}),
      jams::ConfigException);

  reset_constrained_mc_globals();
  EXPECT_THROW(
      make_programmatic_spiral_solver({0.25, 0.0, 0.0}, {0.0, infinity, 1.0}),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, ChecksPeriodicSpinSpiralCommensurability) {
  for (const auto wavevector_component : {0.25, -0.25}) {
    SCOPED_TRACE(wavevector_component);
    EXPECT_NO_THROW(make_spiral_solver(
        "magnetisation", 90.0, 0.0, true,
        {wavevector_component, 0.0, 0.0}, {0.0, 0.0, 1.0}));
    reset_constrained_mc_globals();
  }

  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, true,
          {0.2, 0.0, 0.0}, {0.0, 0.0, 1.0}),
      jams::ConfigException);

  reset_constrained_mc_globals();
  EXPECT_NO_THROW(make_spiral_solver(
      "magnetisation", 90.0, 0.0, true,
      {0.2, 0.0, 0.0}, {0.0, 0.0, 1.0}, {4, 2, 1}, {false, true, true}));
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralAlignsPlaneMagnetisations) {
  const jams::Vec<double, 3> wavevector = {0.25, 0.0, 0.0};
  const jams::Vec<double, 3> axis = {0.0, 0.0, 1.0};
  auto solver = make_spiral_solver(
      "magnetisation", 90.0, 0.0, true, wavevector, {0.0, 0.0, 2.0});

  const jams::Vec<double, 3> reference = {1.0, 0.0, 0.0};
  for (auto plane = 0; plane < 4; ++plane) {
    const auto expected = rotation_matrix_from_axis_angle(
        axis, kTwoPi * wavevector[0] * static_cast<double>(plane)) * reference;
    expect_direction(constrained_mc_plane_total(0, plane, false), expected);
  }
  expect_unit_spins();
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralSupportsTransformedMomentsAndArbitraryAxis) {
  auto solver = make_spiral_solver(
      "material_transform", 0.0, 0.0, true,
      {0.0, 0.5, 0.0}, {2.0, 0.0, 0.0}, {2, 2, 2});

  expect_direction(constrained_mc_plane_total(1, 0, true), {0.0, 0.0, 1.0});
  expect_direction(constrained_mc_plane_total(1, 1, true), {0.0, 0.0, -1.0});
  expect_unit_spins();
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralConstrainsPlaneRatherThanUnitCells) {
  const auto prepare_plane_aggregates = [] {
    constexpr double cell_offset_angle = 0.2;
    for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      const auto cell = globals::lattice->cell_offset(spin_index);
      const jams::Vec<double, 3> target = {
          cell[0] % 2 == 0 ? 1.0 : -1.0, 0.0, 0.0};
      const double offset = cell[1] == 0 ? cell_offset_angle : -cell_offset_angle;
      jams::montecarlo::set_spin(
          spin_index, rotation_matrix_z(offset) * target);
    }
  };

  EXPECT_NO_THROW(make_spiral_solver(
      "magnetisation", 90.0, 0.0, false,
      {0.5, 0.0, 0.0}, {0.0, 0.0, 1.0}, {4, 2, 1}, {true, true, true},
      std::nullopt, prepare_plane_aggregates));

  const auto cell_vector = constrained_mc_cell_total({0, 0, 0}, false);
  const double cell_angular_error = std::atan2(
      jams::norm(jams::cross(cell_vector, jams::Vec<double, 3>{1.0, 0.0, 0.0})),
      jams::dot(cell_vector, jams::Vec<double, 3>{1.0, 0.0, 0.0}));
  EXPECT_GT(cell_angular_error, 0.1);
}

TEST_F(ConstrainedMCSolverConstraintTest, EqualDirectionPlanesRemainSeparateConstraints) {
  const auto prepare_opposite_plane_errors = [] {
    constexpr double plane_error = 0.1;
    for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      const auto cell = globals::lattice->cell_offset(spin_index);
      const jams::Vec<double, 3> target = {
          cell[0] % 2 == 0 ? 1.0 : -1.0, 0.0, 0.0};
      double offset = 0.0;
      if (cell[0] == 0) {
        offset = plane_error;
      } else if (cell[0] == 2) {
        offset = -plane_error;
      }
      jams::montecarlo::set_spin(
          spin_index, rotation_matrix_z(offset) * target);
    }
  };

  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, false,
          {0.5, 0.0, 0.0}, {0.0, 0.0, 1.0}, {4, 2, 1}, {true, true, true},
          std::nullopt, prepare_opposite_plane_errors),
      std::runtime_error);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsPlaneWithFewerThanTwoMagneticSpins) {
  const auto leave_one_magnetic_spin_per_plane = [] {
    for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      if (globals::lattice->lattice_site_basis_index(spin_index) == 1) {
        globals::mus(spin_index) = 0.0;
        globals::inv_mus(spin_index) = 0.0;
      }
    }
  };

  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, true,
          {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, {4, 1, 1}, {true, true, true},
          std::nullopt, leave_one_magnetic_spin_per_plane),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsZeroSpinSpiralPlaneVector) {
  const auto make_plane_vectors_zero = [] {
    for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      globals::mus(spin_index) = 1.0;
      globals::inv_mus(spin_index) = 1.0;
      jams::montecarlo::set_spin(
          spin_index,
          globals::lattice->lattice_site_basis_index(spin_index) == 0
              ? jams::Vec<double, 3>{1.0, 0.0, 0.0}
              : jams::Vec<double, 3>{-1.0, 0.0, 0.0});
    }
  };

  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, true,
          {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, {4, 1, 1}, {true, true, true},
          std::nullopt, make_plane_vectors_zero),
      std::runtime_error);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsNonFiniteSpinSpiralPlaneVector) {
  const auto make_plane_vector_non_finite = [] {
    globals::s(0, 0) = std::numeric_limits<double>::infinity();
  };

  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, true,
          {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, {4, 1, 1}, {true, true, true},
          std::nullopt, make_plane_vector_non_finite),
      std::runtime_error);
}

TEST_F(ConstrainedMCSolverConstraintTest, DefaultToleranceAcceptsRoundoffScaleDirectionalErrors) {
  for (const auto angular_error_degrees : {1.2e-8, 5.0e-7}) {
    SCOPED_TRACE(angular_error_degrees);
    reset_constrained_mc_globals();
    EXPECT_NO_THROW(make_solver(
        "material_transform", 90.0, angular_error_degrees, false));
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, DefaultToleranceRejectsLargerDirectionalError) {
  try {
    auto solver = make_solver("material_transform", 90.0, 2.0e-6, false);
    FAIL() << "expected constraint validation to fail";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("angular difference"), std::string::npos);
    EXPECT_NE(message.find("exceeds tolerance"), std::string::npos);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, ConfiguredLooseToleranceAcceptsLargerDirectionalError) {
  EXPECT_NO_THROW(make_solver(
      "material_transform", 90.0, 5.0e-5, false, 1.0e-4));
}

TEST_F(ConstrainedMCSolverConstraintTest, ConfiguredStrictToleranceRejectsSmallDirectionalError) {
  EXPECT_THROW(
      make_solver("material_transform", 90.0, 5.0e-7, false, 1.0e-8),
      std::runtime_error);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsInvalidConstraintTolerances) {
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto invalid_tolerance : {0.0, -1.0, 180.000001, infinity, nan}) {
    SCOPED_TRACE(invalid_tolerance);
    reset_constrained_mc_globals();
    EXPECT_THROW(
        make_solver("material_transform", 90.0, 0.0, false, invalid_tolerance),
        jams::ConfigException);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, AcceptsEquivalentDirectionAtAzimuthBranchCut) {
  auto solver = make_solver("material_transform", 90.0, -180.0, true);
  expect_direction(constrained_mc_total(true), {-1.0, 0.0, 0.0});
}

TEST_F(ConstrainedMCSolverConstraintTest, PolarDirectionDoesNotDependOnAzimuth) {
  auto solver = make_solver("material_transform", 0.0, 123.0, true);
  expect_direction(constrained_mc_total(true), {0.0, 0.0, 1.0});
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsZeroLengthConstraintVector) {
  const auto make_transformed_total_zero = [] {
    globals::mus(0) = 1.0;
    globals::inv_mus(0) = 1.0;
    globals::mus(1) = 1.0;
    globals::inv_mus(1) = 1.0;
  };

  try {
    auto solver = make_solver(
        "material_transform",
        90.0,
        0.0,
        false,
        std::nullopt,
        make_transformed_total_zero);
    FAIL() << "expected an undefined constraint direction";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string(error.what()).find("constraint direction is undefined"),
        std::string::npos);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsNonFiniteConstraintVector) {
  const auto make_total_non_finite = [] {
    globals::s(0, 0) = std::numeric_limits<double>::infinity();
  };

  try {
    auto solver = make_solver(
        "material_transform",
        90.0,
        0.0,
        false,
        std::nullopt,
        make_total_non_finite);
    FAIL() << "expected an undefined constraint direction";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string(error.what()).find("constraint direction is undefined"),
        std::string::npos);
  }
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

TEST_F(ConstrainedMCSolverConstraintTest, PairMovesPreserveSpinSpiralPlaneMagnetisation) {
  exercise_spiral_pair_moves("magnetisation", false);
}

TEST_F(ConstrainedMCSolverConstraintTest, PairMovesPreserveSpinSpiralPlaneTransformedMoments) {
  exercise_spiral_pair_moves("material_transform", true);
}

#endif  // JAMS_TEST_SOLVERS_TEST_CPU_MONTE_CARLO_CONSTRAINED_H
