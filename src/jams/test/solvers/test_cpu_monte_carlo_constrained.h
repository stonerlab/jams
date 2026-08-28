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
    const std::optional<jams::Vec<double, 3>>& spiral_background = std::nullopt,
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
  if (spiral_background.has_value()) {
    auto background = *spiral_background;
    jams::Vec<double, 3> polarisation = {1.0, 0.0, 0.0};
    double amplitude_degrees = 90.0;
    const auto background_norm = jams::norm(background);
    if (jams::is_finite(background)
        && std::isfinite(background_norm)
        && background_norm != 0.0) {
      background /= background_norm;
      const auto reference = jams::spherical_to_cartesian_vector(
          1.0, deg_to_rad(theta), deg_to_rad(phi));
      const auto cosine_amplitude = std::clamp(
          jams::dot(background, reference), -1.0, 1.0);
      polarisation = reference - cosine_amplitude * background;
      const auto polarisation_norm = jams::norm(polarisation);
      if (polarisation_norm == 0.0) {
        polarisation = std::abs(background[2]) < 0.9
            ? jams::normalize(jams::cross(background, jams::Vec<double, 3>{0.0, 0.0, 1.0}))
            : jams::normalize(jams::cross(background, jams::Vec<double, 3>{0.0, 1.0, 0.0}));
      } else {
        polarisation /= polarisation_norm;
      }
      amplitude_degrees = rad_to_deg(std::acos(cosine_amplitude));
    } else {
      polarisation = {0.0, 1.0, 0.0};
    }
    config << "      cmc_spiral_profile = \"circular\";\n"
           << "      cmc_spiral_background = ["
           << background[0] << ", " << background[1] << ", "
           << background[2] << "];\n"
           << "      cmc_spiral_polarisation = ["
           << polarisation[0] << ", " << polarisation[1] << ", "
           << polarisation[2] << "];\n"
           << "      cmc_spiral_amplitude = " << amplitude_degrees << ";\n";
  }
  if (lowercase(constraint_mode) != "spin_spiral") {
    config << "      cmc_constraint_theta = " << theta << ";\n"
           << "      cmc_constraint_phi = " << phi << ";\n";
  }
  config << R"(    };

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
      const jams::Vec<double, 3>& background,
      const std::optional<int> propagation_direction = std::nullopt,
      const jams::Vec<int, 3>& lattice_size = {4, 2, 1},
      const std::string& profile = "circular",
      const std::optional<jams::Vec<double, 3>>& polarisation = std::nullopt,
      const double amplitude_degrees = 90.0,
      const double phase_degrees = 0.0) {
    globals::config = std::make_unique<libconfig::Config>();
    const auto config_text = constrained_mc_config(
        "magnetisation", 90.0, 0.0, true, "", std::nullopt, std::nullopt,
        lattice_size, {true, true, true});
    globals::config->readString(config_text.c_str());
    auto& solver_settings = globals::config->lookup("solver");
    solver_settings.remove("cmc_constraint_theta");
    solver_settings.remove("cmc_constraint_phi");
    solver_settings.add("cmc_constraint_mode", libconfig::Setting::TypeString)
        = "spin_spiral";
    add_vec_setting(solver_settings, "cmc_spiral_wavevector", wavevector);
    solver_settings.add("cmc_spiral_profile", libconfig::Setting::TypeString)
        = profile;
    add_vec_setting(solver_settings, "cmc_spiral_background", background);
    auto configured_polarisation = polarisation.value_or(
        jams::Vec<double, 3>{1.0, 0.0, 0.0});
    add_vec_setting(
        solver_settings, "cmc_spiral_polarisation", configured_polarisation);
    solver_settings.add("cmc_spiral_amplitude", libconfig::Setting::TypeFloat)
        = amplitude_degrees;
    solver_settings.add("cmc_spiral_phase", libconfig::Setting::TypeFloat)
        = phase_degrees;
    if (propagation_direction.has_value()) {
      solver_settings.add(
          "cmc_spiral_propagation_direction", libconfig::Setting::TypeInt)
          = *propagation_direction;
    }
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    return std::make_unique<ConstrainedMCSolver>(solver_settings);
  }

  std::unique_ptr<ConstrainedMCSolver> make_profile_spiral_solver(
      const std::string& constraint_type,
      const std::string& profile,
      const jams::Vec<double, 3>& wavevector,
      const jams::Vec<double, 3>& background,
      const jams::Vec<double, 3>& polarisation,
      const double amplitude_degrees,
      const double phase_degrees = 0.0,
      const jams::Vec<int, 3>& lattice_size = {4, 2, 1},
      const std::optional<int> propagation_direction = std::nullopt) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(
        constrained_mc_config(
            constraint_type, 90.0, 0.0, true, "", std::nullopt,
            std::nullopt, lattice_size, {true, true, true}).c_str());
    auto& solver_settings = globals::config->lookup("solver");
    configure_spin_spiral(
        solver_settings, profile, wavevector, background, polarisation,
        amplitude_degrees, phase_degrees, propagation_direction);
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    return std::make_unique<ConstrainedMCSolver>(solver_settings);
  }

  std::unique_ptr<ConstrainedMCSolver> make_configured_solver(
      const double temperature,
      const std::function<void(libconfig::Setting&)>& configure_solver,
      const bool register_physics = true) {
    globals::config = std::make_unique<libconfig::Config>();
    globals::config->readString(
        constrained_mc_config("magnetisation", 90.0, 0.0, true).c_str());
    globals::config->lookup("physics.temperature") = temperature;
    auto& solver_settings = globals::config->lookup("solver");
    configure_solver(solver_settings);
    globals::lattice = new Lattice();
    globals::lattice->init_from_config(*globals::config);
    auto solver = std::make_unique<ConstrainedMCSolver>(solver_settings);
    if (register_physics) {
      solver->register_physics_module(
          Physics::create(globals::config->lookup("physics")));
    }
    return solver;
  }

  void configure_spin_spiral(
      libconfig::Setting& solver_settings,
      const std::string& profile = "circular",
      const jams::Vec<double, 3>& wavevector = {0.25, 0.0, 0.0},
      const jams::Vec<double, 3>& background = {0.0, 0.0, 1.0},
      const jams::Vec<double, 3>& polarisation = {1.0, 0.0, 0.0},
      const double amplitude_degrees = 30.0,
      const double phase_degrees = 0.0,
      const std::optional<int> propagation_direction = std::nullopt) {
    solver_settings.remove("cmc_constraint_theta");
    solver_settings.remove("cmc_constraint_phi");
    solver_settings.add("cmc_constraint_mode", libconfig::Setting::TypeString)
        = "spin_spiral";
    solver_settings.add("cmc_spiral_profile", libconfig::Setting::TypeString)
        = profile;
    add_vec_setting(solver_settings, "cmc_spiral_wavevector", wavevector);
    add_vec_setting(solver_settings, "cmc_spiral_background", background);
    add_vec_setting(solver_settings, "cmc_spiral_polarisation", polarisation);
    solver_settings.add("cmc_spiral_amplitude", libconfig::Setting::TypeFloat)
        = amplitude_degrees;
    solver_settings.add("cmc_spiral_phase", libconfig::Setting::TypeFloat)
        = phase_degrees;
    if (propagation_direction.has_value()) {
      solver_settings.add(
          "cmc_spiral_propagation_direction", libconfig::Setting::TypeInt)
          = *propagation_direction;
    }
  }

  void add_move_angle_adaptation(
      libconfig::Setting& solver_settings,
      const std::optional<double> target_acceptance = 0.1,
      const std::optional<int> interval_steps = 10,
      const std::optional<double> gain = 0.5,
      const std::optional<double> min_sigma = 1.0e-6,
      const std::optional<double> max_sigma = 0.1,
      const std::optional<int> burn_in_steps = std::nullopt,
      const bool enabled = true) {
    auto& adaptation = solver_settings.add(
        "move_angle_adaptation", libconfig::Setting::TypeGroup);
    adaptation.add("enabled", libconfig::Setting::TypeBoolean) = enabled;
    if (target_acceptance.has_value()) {
      adaptation.add("target_acceptance", libconfig::Setting::TypeFloat)
          = *target_acceptance;
    }
    if (interval_steps.has_value()) {
      adaptation.add("interval_steps", libconfig::Setting::TypeInt)
          = *interval_steps;
    }
    if (gain.has_value()) {
      adaptation.add("gain", libconfig::Setting::TypeFloat) = *gain;
    }
    if (min_sigma.has_value()) {
      adaptation.add("min_sigma", libconfig::Setting::TypeFloat) = *min_sigma;
    }
    if (max_sigma.has_value()) {
      adaptation.add("max_sigma", libconfig::Setting::TypeFloat) = *max_sigma;
    }
    if (burn_in_steps.has_value()) {
      adaptation.add("burn_in_steps", libconfig::Setting::TypeInt)
          = *burn_in_steps;
    }
  }

  std::string perform_adaptation_update(
      ConstrainedMCSolver& solver,
      const int step,
      const unsigned long long attempted,
      const unsigned long long accepted) {
    solver.iteration_ = step;
    solver.move_angle_adaptation_attempted_ = attempted;
    solver.move_angle_adaptation_accepted_ = accepted;
    std::ostringstream output;
    solver.update_move_angle_adaptation(output);
    return output.str();
  }

  std::size_t constraint_plane_count(const ConstrainedMCSolver& solver) const {
    return solver.constraint_planes_.size();
  }

  int constraint_plane_coordinate(
      const ConstrainedMCSolver& solver, const std::size_t plane) const {
    return solver.constraint_planes_[plane].coordinate;
  }

  const std::vector<int>& constraint_plane_spins(
      const ConstrainedMCSolver& solver, const std::size_t plane) const {
    return solver.constraint_planes_[plane].spins;
  }

  jams::Vec<double, 3> constraint_plane_target(
      const ConstrainedMCSolver& solver, const std::size_t plane) const {
    return solver.constraint_planes_[plane].target_direction;
  }

  int spin_constraint_group(
      const ConstrainedMCSolver& solver, const int spin) const {
    return solver.spin_constraint_group_[spin];
  }

  double move_angle_sigma(const ConstrainedMCSolver& solver) const {
    return solver.move_angle_sigma_;
  }

  bool move_angle_adaptation_frozen(const ConstrainedMCSolver& solver) const {
    return solver.move_angle_adaptation_frozen_;
  }

  std::pair<unsigned long long, unsigned long long> adaptation_counters(
      const ConstrainedMCSolver& solver) const {
    return {solver.move_angle_adaptation_attempted_,
            solver.move_angle_adaptation_accepted_};
  }

  void set_non_angle_statistics(
      ConstrainedMCSolver& solver,
      const unsigned long long uniform,
      const unsigned long long reflection) {
    solver.move_running_acceptance_count_uniform_ = uniform;
    solver.move_running_acceptance_count_reflection_ = reflection;
  }

  void set_solver_temperature(ConstrainedMCSolver& solver, const double temperature) {
    solver.physics_module_->set_temperature(temperature);
  }

  std::string initialization_output(ConstrainedMCSolver& solver) {
    std::ostringstream output;
    solver.output_initialization_info(output);
    return output.str();
  }

  std::vector<jams::Vec<double, 3>> run_seeded_solver(
      ConstrainedMCSolver& solver,
      const unsigned seed,
      const int steps) {
    jams::instance().random_generator().seed(seed);
    std::vector<jams::Vec<double, 3>> trajectory;
    trajectory.reserve(steps);
    for (auto step = 0; step < steps; ++step) {
      solver.run();
      trajectory.push_back(jams::montecarlo::get_spin(0));
    }
    return trajectory;
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
      const bool transformed,
      const std::string& profile = "circular") {
    const jams::Vec<double, 3> wavevector = {0.25, 0.0, 0.0};
    auto solver = make_profile_spiral_solver(
        constraint_type, profile, wavevector, {0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0}, 30.0);
    solver->register_physics_module(Physics::create(globals::config->lookup("physics")));

    std::vector<jams::Vec<double, 3>> expected_targets;
    for (auto plane = 0; plane < 4; ++plane) {
      expected_targets.push_back(constraint_plane_target(*solver, plane));
    }

    jams::instance().random_generator().seed(24680u);
    for (auto step = 0; step < 128; ++step) {
      ASSERT_NO_THROW(solver->run());
    }

    double maximum_cell_angular_error = 0.0;
    for (auto plane = 0; plane < 4; ++plane) {
      expect_direction(
          constrained_mc_plane_total(0, plane, transformed),
          expected_targets[plane]);

      for (auto transverse_cell = 0; transverse_cell < 2; ++transverse_cell) {
        const auto cell_vector = constrained_mc_cell_total(
            {plane, transverse_cell, 0}, transformed);
        const double angular_error = std::atan2(
            jams::norm(jams::cross(cell_vector, expected_targets[plane])),
            jams::dot(cell_vector, expected_targets[plane]));
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

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralRequiresModulationSettings) {
  const jams::Vec<double, 3> wavevector = {0.25, 0.0, 0.0};
  const jams::Vec<double, 3> background = {0.0, 0.0, 1.0};
  const auto missing_background = constrained_mc_config(
      "magnetisation", 90.0, 0.0, true, "spin_spiral", wavevector,
      std::nullopt, {4, 2, 1});
  EXPECT_THROW(make_solver_from_config(missing_background), jams::ConfigException);

  reset_constrained_mc_globals();
  const auto missing_wavevector = constrained_mc_config(
      "magnetisation", 90.0, 0.0, true, "spin_spiral", std::nullopt,
      background, {4, 2, 1});
  EXPECT_THROW(make_solver_from_config(missing_wavevector), jams::ConfigException);

  for (const auto* setting : {
           "cmc_spiral_profile",
           "cmc_spiral_background",
           "cmc_spiral_polarisation",
           "cmc_spiral_amplitude"}) {
    reset_constrained_mc_globals();
    EXPECT_THROW(
        make_configured_solver(0.0, [this, setting](libconfig::Setting& settings) {
          configure_spin_spiral(settings);
          settings.remove(setting);
        }),
        jams::ConfigException);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, GlobalModeRejectsSpiralSettings) {
  const auto config_text = constrained_mc_config(
      "magnetisation", 90.0, 0.0, true, "global",
      jams::Vec<double, 3>{0.25, 0.0, 0.0},
      jams::Vec<double, 3>{0.0, 0.0, 1.0}, {4, 2, 1});
  EXPECT_THROW(make_solver_from_config(config_text), jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralProfileIsCaseInsensitive) {
  EXPECT_NO_THROW(make_programmatic_spiral_solver(
      {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt, {4, 2, 1},
      "LiNeAr", jams::Vec<double, 3>{1.0, 0.0, 0.0}, 5.0));
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsUnknownSpinSpiralProfile) {
  EXPECT_THROW(
      make_programmatic_spiral_solver(
          {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt,
          {4, 2, 1}, "elliptical"),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsReplacedSpinSpiralSettings) {
  for (const auto* setting : {
           "cmc_constraint_theta", "cmc_constraint_phi", "cmc_spiral_axis"}) {
    reset_constrained_mc_globals();
    EXPECT_THROW(
        make_configured_solver(0.0, [this, setting](libconfig::Setting& settings) {
          configure_spin_spiral(settings);
          if (std::string(setting) == "cmc_spiral_axis") {
            add_vec_setting(settings, setting, {0.0, 0.0, 1.0});
          } else {
            settings.add(setting, libconfig::Setting::TypeFloat) = 0.0;
          }
        }),
        jams::ConfigException);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, NormalisesSpinSpiralFrameVectors) {
  auto solver = make_programmatic_spiral_solver(
      {0.25, 0.0, 0.0}, {0.0, 0.0, 2.0}, std::nullopt, {4, 2, 1},
      "linear", jams::Vec<double, 3>{3.0, 0.0, 0.0}, 30.0);
  expect_direction(
      constraint_plane_target(*solver, 0),
      {0.5, 0.0, std::sqrt(3.0) / 2.0});
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsInvalidSpinSpiralFrameVectors) {
  for (const auto& [background, polarisation] : {
           std::pair{jams::Vec<double, 3>{0.0, 0.0, 0.0}, jams::Vec<double, 3>{1.0, 0.0, 0.0}},
           std::pair{jams::Vec<double, 3>{0.0, 0.0, 1.0}, jams::Vec<double, 3>{0.0, 0.0, 0.0}},
           std::pair{jams::Vec<double, 3>{0.0, 0.0, 1.0}, jams::Vec<double, 3>{1.0, 0.0, 1.0}}}) {
    reset_constrained_mc_globals();
    EXPECT_THROW(
        make_programmatic_spiral_solver(
            {0.25, 0.0, 0.0}, background, std::nullopt, {4, 2, 1},
            "linear", polarisation, 5.0),
        jams::ConfigException);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, ValidatesSpinSpiralAmplitudeByProfile) {
  const auto infinity = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto amplitude : {-1.0, 181.0, infinity, nan}) {
    reset_constrained_mc_globals();
    EXPECT_THROW(
        make_programmatic_spiral_solver(
            {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt,
            {4, 2, 1}, "circular", std::nullopt, amplitude),
        jams::ConfigException);
  }
  reset_constrained_mc_globals();
  EXPECT_NO_THROW(make_programmatic_spiral_solver(
      {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt, {4, 2, 1},
      "circular", std::nullopt, 180.0));

  for (const auto amplitude : {-1.0, 90.0, infinity, nan}) {
    reset_constrained_mc_globals();
    EXPECT_THROW(
        make_programmatic_spiral_solver(
            {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt,
            {4, 2, 1}, "linear", std::nullopt, amplitude),
        jams::ConfigException);
  }
  reset_constrained_mc_globals();
  EXPECT_NO_THROW(make_programmatic_spiral_solver(
      {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt, {4, 2, 1},
      "linear", std::nullopt, 0.0));
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsNonFiniteSpinSpiralPhase) {
  EXPECT_THROW(
      make_programmatic_spiral_solver(
          {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt,
          {4, 2, 1}, "linear", std::nullopt, 5.0,
          std::numeric_limits<double>::infinity()),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsInvalidSpinSpiralVectors) {
  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, true,
          {0.25, 0.25, 0.0}, {0.0, 0.0, 1.0}),
      jams::ConfigException);

  reset_constrained_mc_globals();
  EXPECT_THROW(
      make_spiral_solver(
          "magnetisation", 90.0, 0.0, true, {0.25, 0.0, 0.0}, {0.0, 0.0, 0.0}),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, ZeroWavevectorRequiresPropagationDirection) {
  EXPECT_THROW(
      make_programmatic_spiral_solver(
          {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, ZeroWavevectorAcceptsEveryPropagationDirection) {
  const jams::Vec<int, 3> lattice_size = {2, 3, 4};
  for (const auto* profile : {"circular", "linear"}) {
    for (auto direction = 0; direction < 3; ++direction) {
      SCOPED_TRACE(profile);
      SCOPED_TRACE(direction);
      auto solver = make_programmatic_spiral_solver(
          {0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}, direction, lattice_size,
          profile, jams::Vec<double, 3>{1.0, 0.0, 0.0}, 30.0);

      ASSERT_EQ(constraint_plane_count(*solver), lattice_size[direction]);
      const auto spins_per_plane =
          2 * lattice_size[(direction + 1) % 3]
            * lattice_size[(direction + 2) % 3];
      for (auto plane = 0; plane < constraint_plane_count(*solver); ++plane) {
        const auto coordinate = constraint_plane_coordinate(*solver, plane);
        const auto& spins = constraint_plane_spins(*solver, plane);
        EXPECT_EQ(spins.size(), spins_per_plane);
        for (const auto spin : spins) {
          EXPECT_EQ(
              globals::lattice->cell_offset(spin)[direction], coordinate);
          EXPECT_EQ(spin_constraint_group(*solver, spin), plane);
        }
        expect_direction(
            constraint_plane_target(*solver, plane),
            constraint_plane_target(*solver, 0));
      }

      solver.reset();
      reset_constrained_mc_globals();
    }
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, ZeroWavevectorPartnersPreserveEveryPlaneConstraint) {
  for (const auto* profile : {"circular", "linear"}) {
    SCOPED_TRACE(profile);
    auto solver = make_programmatic_spiral_solver(
        {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, 0, {4, 2, 1},
        profile, jams::Vec<double, 3>{1.0, 0.0, 0.0}, 30.0);
    const auto expected = constraint_plane_target(*solver, 0);
    solver->register_physics_module(
        Physics::create(globals::config->lookup("physics")));

    jams::instance().random_generator().seed(97531u);
    for (auto step = 0; step < 128; ++step) {
      ASSERT_NO_THROW(solver->run());
    }

    for (auto plane = 0; plane < 4; ++plane) {
      expect_direction(
          constrained_mc_plane_total(0, plane, false), expected);
    }
    solver.reset();
    reset_constrained_mc_globals();
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, ZeroWavevectorMatchesReciprocalLatticeVectorWorkaround) {
  const auto run_case = [this](
      const double wavevector_component, const std::string& profile) {
    auto solver = make_programmatic_spiral_solver(
        {wavevector_component, 0.0, 0.0}, {0.0, 0.0, 1.0}, 0,
        {4, 2, 1}, profile, jams::Vec<double, 3>{1.0, 0.0, 0.0}, 30.0);
    solver->register_physics_module(
        Physics::create(globals::config->lookup("physics")));
    jams::instance().random_generator().seed(86420u);
    for (auto step = 0; step < 64; ++step) {
      solver->run();
    }
    std::vector<jams::Vec<double, 3>> spins(globals::num_spins);
    for (auto spin = 0; spin < globals::num_spins; ++spin) {
      spins[spin] = jams::montecarlo::get_spin(spin);
    }
    solver.reset();
    reset_constrained_mc_globals();
    return spins;
  };

  for (const auto* profile : {"circular", "linear"}) {
    SCOPED_TRACE(profile);
    const auto zero_wavevector_spins = run_case(0.0, profile);
    const auto reciprocal_wavevector_spins = run_case(1.0, profile);
    ASSERT_EQ(zero_wavevector_spins.size(), reciprocal_wavevector_spins.size());
    for (auto spin = 0; spin < zero_wavevector_spins.size(); ++spin) {
      for (auto component = 0; component < 3; ++component) {
        EXPECT_NEAR(
            zero_wavevector_spins[spin][component],
            reciprocal_wavevector_spins[spin][component], 1.0e-10);
      }
    }
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, NonzeroWavevectorAcceptsAbsentOrMatchingDirection) {
  EXPECT_NO_THROW(make_programmatic_spiral_solver(
      {0.0, 0.5, 0.0}, {0.0, 0.0, 1.0}));

  reset_constrained_mc_globals();
  EXPECT_NO_THROW(make_programmatic_spiral_solver(
      {0.0, 0.5, 0.0}, {0.0, 0.0, 1.0}, 1));
}

TEST_F(ConstrainedMCSolverConstraintTest, NonzeroWavevectorRejectsConflictingDirection) {
  EXPECT_THROW(
      make_programmatic_spiral_solver(
          {0.0, 0.5, 0.0}, {0.0, 0.0, 1.0}, 2),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsInvalidPropagationDirections) {
  for (const auto direction : {-1, 3}) {
    SCOPED_TRACE(direction);
    EXPECT_THROW(
        make_programmatic_spiral_solver(
            {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, direction),
        jams::ConfigException);
    reset_constrained_mc_globals();
  }

  EXPECT_THROW(
      make_configured_solver(0.0, [this](libconfig::Setting& settings) {
        configure_spin_spiral(
            settings, "circular", {0.0, 0.0, 0.0},
            {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 30.0);
        settings.add(
            "cmc_spiral_propagation_direction", libconfig::Setting::TypeFloat)
            = 1.0;
      }),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, GlobalModeRejectsPropagationDirection) {
  EXPECT_THROW(
      make_configured_solver(0.0, [](libconfig::Setting& settings) {
        settings.add(
            "cmc_spiral_propagation_direction", libconfig::Setting::TypeInt) = 0;
      }),
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

  reset_constrained_mc_globals();
  EXPECT_THROW(
      make_programmatic_spiral_solver(
          {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt,
          {4, 2, 1}, "linear",
          jams::Vec<double, 3>{infinity, 0.0, 0.0}, 5.0),
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

TEST_F(ConstrainedMCSolverConstraintTest, CircularProfileUsesTiltedLocalFrame) {
  const auto inverse_sqrt_two = 1.0 / std::sqrt(2.0);
  const jams::Vec<double, 3> background = {
      inverse_sqrt_two, 0.0, inverse_sqrt_two};
  const jams::Vec<double, 3> polarisation = {
      inverse_sqrt_two, 0.0, -inverse_sqrt_two};
  constexpr double amplitude_degrees = 12.0;
  constexpr double phase_degrees = 30.0;
  auto solver = make_programmatic_spiral_solver(
      {0.25, 0.0, 0.0}, background, std::nullopt, {4, 2, 1},
      "circular", polarisation, amplitude_degrees, phase_degrees);

  const auto amplitude = deg_to_rad(amplitude_degrees);
  for (auto plane = 0; plane < 4; ++plane) {
    const auto phase = deg_to_rad(phase_degrees) + 0.5 * kPi * plane;
    const auto expected = std::cos(amplitude) * background
        + std::sin(amplitude)
            * (std::cos(phase) * polarisation
               + std::sin(phase) * jams::cross(background, polarisation));
    expect_direction(constraint_plane_target(*solver, plane), expected);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, LinearProfileFollowsStandingWavePhases) {
  auto solver = make_programmatic_spiral_solver(
      {0.25, 0.0, 0.0}, {0.0, 0.0, 1.0}, std::nullopt, {4, 2, 1},
      "linear", jams::Vec<double, 3>{1.0, 0.0, 0.0}, 30.0);

  expect_direction(
      constraint_plane_target(*solver, 0),
      {0.5, 0.0, std::sqrt(3.0) / 2.0});
  expect_direction(constraint_plane_target(*solver, 1), {0.0, 0.0, 1.0});
  expect_direction(
      constraint_plane_target(*solver, 2),
      {-0.5, 0.0, std::sqrt(3.0) / 2.0});
  expect_direction(constraint_plane_target(*solver, 3), {0.0, 0.0, 1.0});
}

TEST_F(ConstrainedMCSolverConstraintTest, LinearProfileAcceptsTensorProbePolarisations) {
  const auto inverse_sqrt_two = 1.0 / std::sqrt(2.0);
  const std::vector<jams::Vec<double, 3>> polarisations = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {inverse_sqrt_two, inverse_sqrt_two, 0.0},
      {inverse_sqrt_two, -inverse_sqrt_two, 0.0}};
  const auto amplitude = deg_to_rad(5.0);
  for (const auto& polarisation : polarisations) {
    reset_constrained_mc_globals();
    auto solver = make_programmatic_spiral_solver(
        {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, 0, {4, 2, 1},
        "linear", polarisation, 5.0);
    const auto expected =
        std::cos(amplitude) * jams::Vec<double, 3>{0.0, 0.0, 1.0}
        + std::sin(amplitude) * polarisation;
    expect_direction(constraint_plane_target(*solver, 0), expected);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralPhaseOffsetsQZeroTarget) {
  auto solver = make_programmatic_spiral_solver(
      {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, 0, {4, 2, 1},
      "linear", jams::Vec<double, 3>{1.0, 0.0, 0.0}, 30.0, 60.0);
  const auto expected = jams::normalize(
      std::cos(deg_to_rad(30.0)) * jams::Vec<double, 3>{0.0, 0.0, 1.0}
      + std::sin(deg_to_rad(30.0)) * std::cos(deg_to_rad(60.0))
          * jams::Vec<double, 3>{1.0, 0.0, 0.0});
  for (auto plane = 0; plane < 4; ++plane) {
    expect_direction(constraint_plane_target(*solver, plane), expected);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, LogsSpinSpiralPolarisationGeometry) {
  auto solver = make_programmatic_spiral_solver(
      {0.25, 0.0, 0.0}, {0.0, 0.0, 2.0}, std::nullopt, {4, 2, 1},
      "linear", jams::Vec<double, 3>{3.0, 0.0, 0.0}, 5.0, 15.0);
  const auto output = initialization_output(*solver);
  EXPECT_NE(output.find("spiral profile linear"), std::string::npos);
  EXPECT_NE(output.find("spiral background"), std::string::npos);
  EXPECT_NE(output.find("spiral polarisation"), std::string::npos);
  EXPECT_NE(output.find("spiral amplitude (deg) 5"), std::string::npos);
  EXPECT_NE(output.find("spiral phase (deg) 15"), std::string::npos);
  EXPECT_NE(output.find("spiral reference target"), std::string::npos);
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralAlignsPlaneMagnetisations) {
  const jams::Vec<double, 3> wavevector = {0.25, 0.0, 0.0};
  for (const auto* profile : {"circular", "linear"}) {
    SCOPED_TRACE(profile);
    auto solver = make_profile_spiral_solver(
        "magnetisation", profile, wavevector, {0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0}, 30.0);

    for (auto plane = 0; plane < 4; ++plane) {
      expect_direction(
          constrained_mc_plane_total(0, plane, false),
          constraint_plane_target(*solver, plane));
    }
    expect_unit_spins();
    solver.reset();
    reset_constrained_mc_globals();
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, SpinSpiralSupportsTransformedMomentsForBothProfiles) {
  for (const auto* profile : {"circular", "linear"}) {
    SCOPED_TRACE(profile);
    auto solver = make_profile_spiral_solver(
        "material_transform", profile, {0.0, 0.5, 0.0},
        {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, 30.0, 0.0, {2, 2, 2});

    for (auto plane = 0; plane < 2; ++plane) {
      expect_direction(
          constrained_mc_plane_total(1, plane, true),
          constraint_plane_target(*solver, plane));
    }
    expect_unit_spins();
    solver.reset();
    reset_constrained_mc_globals();
  }
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

TEST_F(ConstrainedMCSolverConstraintTest, PairMovesPreserveLinearPlaneMagnetisation) {
  exercise_spiral_pair_moves("magnetisation", false, "linear");
}

TEST_F(ConstrainedMCSolverConstraintTest, PairMovesPreserveLinearPlaneTransformedMoments) {
  exercise_spiral_pair_moves("material_transform", true, "linear");
}

TEST_F(ConstrainedMCSolverConstraintTest, DisabledAdaptationPreservesSeededTrajectory) {
  const auto run_case = [this](const bool add_disabled_block) {
    auto solver = make_configured_solver(
        1000000.0, [this, add_disabled_block](libconfig::Setting& settings) {
          settings["output_write_steps"] = 7;
          if (add_disabled_block) {
            add_move_angle_adaptation(
                settings, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, std::nullopt, false);
          }
        });
    const auto trajectory = run_seeded_solver(*solver, 112233u, 32);
    solver.reset();
    reset_constrained_mc_globals();
    return trajectory;
  };

  const auto legacy_trajectory = run_case(false);
  const auto disabled_trajectory = run_case(true);
  ASSERT_EQ(legacy_trajectory.size(), disabled_trajectory.size());
  for (auto step = 0; step < legacy_trajectory.size(); ++step) {
    EXPECT_EQ(legacy_trajectory[step], disabled_trajectory[step]);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationRespondsToAngleAcceptance) {
  auto make_adaptive_solver = [this] {
    return make_configured_solver(0.0, [this](libconfig::Setting& settings) {
      add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
    });
  };

  auto solver = make_adaptive_solver();
  const auto initial_sigma = move_angle_sigma(*solver);
  perform_adaptation_update(*solver, 10, 100, 0);
  EXPECT_LT(move_angle_sigma(*solver), initial_sigma);

  solver.reset();
  reset_constrained_mc_globals();
  solver = make_adaptive_solver();
  perform_adaptation_update(*solver, 10, 100, 100);
  EXPECT_GT(move_angle_sigma(*solver), initial_sigma);

  solver.reset();
  reset_constrained_mc_globals();
  solver = make_adaptive_solver();
  perform_adaptation_update(*solver, 10, 100, 10);
  EXPECT_DOUBLE_EQ(move_angle_sigma(*solver), initial_sigma);
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationClampsSigmaAtBothBounds) {
  auto make_adaptive_solver = [this](const double target_acceptance) {
    return make_configured_solver(0.0, [this, target_acceptance](libconfig::Setting& settings) {
      add_move_angle_adaptation(
          settings, target_acceptance, 10, 100.0, 0.01, 0.1);
    });
  };

  auto solver = make_adaptive_solver(0.9);
  const auto minimum_output = perform_adaptation_update(*solver, 10, 100, 0);
  EXPECT_DOUBLE_EQ(move_angle_sigma(*solver), 0.01);
  EXPECT_NE(minimum_output.find("bound minimum"), std::string::npos);

  solver.reset();
  reset_constrained_mc_globals();
  solver = make_adaptive_solver(0.1);
  const auto maximum_output = perform_adaptation_update(*solver, 10, 100, 100);
  EXPECT_DOUBLE_EQ(move_angle_sigma(*solver), 0.1);
  EXPECT_NE(maximum_output.find("bound maximum"), std::string::npos);
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationUsesOnlyAngleMoveStatistics) {
  auto solver = make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    add_move_angle_adaptation(settings, 0.5, 10, 1.0, 1.0e-6, 0.1);
  });
  const auto initial_sigma = move_angle_sigma(*solver);
  set_non_angle_statistics(*solver, 1000000, 1000000);
  perform_adaptation_update(*solver, 10, 100, 0);
  EXPECT_LT(move_angle_sigma(*solver), initial_sigma);
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationResetsItsIntervalCounters) {
  auto solver = make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
  });
  perform_adaptation_update(*solver, 10, 100, 25);
  EXPECT_EQ(adaptation_counters(*solver), std::make_pair(0ULL, 0ULL));
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationSkipsIntervalsWithoutAngleMoves) {
  auto solver = make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
  });
  const auto initial_sigma = move_angle_sigma(*solver);
  const auto output = perform_adaptation_update(*solver, 10, 0, 0);
  EXPECT_DOUBLE_EQ(move_angle_sigma(*solver), initial_sigma);
  EXPECT_NE(output.find("update skipped (no angle moves attempted)"), std::string::npos);
  EXPECT_EQ(adaptation_counters(*solver), std::make_pair(0ULL, 0ULL));
}

TEST_F(ConstrainedMCSolverConstraintTest, ZeroTemperatureAdaptationAllowsNoBurnIn) {
  EXPECT_NO_THROW(make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
  }));
}

TEST_F(ConstrainedMCSolverConstraintTest, OptionalZeroTemperatureBurnInFreezesSigma) {
  auto solver = make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    settings["output_write_steps"] = 7;
    add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1, 3);
  });
  const auto initial_sigma = move_angle_sigma(*solver);
  for (auto step = 0; step < 3; ++step) {
    solver->run();
  }
  const auto frozen_sigma = move_angle_sigma(*solver);
  EXPECT_LT(frozen_sigma, initial_sigma);
  EXPECT_TRUE(move_angle_adaptation_frozen(*solver));
  for (auto step = 0; step < 10; ++step) {
    solver->run();
    EXPECT_DOUBLE_EQ(move_angle_sigma(*solver), frozen_sigma);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, NonzeroTemperatureAdaptationRequiresBurnIn) {
  try {
    auto solver = make_configured_solver(
        300.0, [this](libconfig::Setting& settings) {
          add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
        });
    FAIL() << "expected missing finite-temperature burn-in to fail";
  } catch (const jams::ConfigException& error) {
    EXPECT_NE(
        std::string(error.what()).find("move_angle_adaptation.burn_in_steps"),
        std::string::npos);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, FiniteTemperatureBurnInFreezesProductionSigma) {
  auto solver = make_configured_solver(300.0, [this](libconfig::Setting& settings) {
    settings["output_write_steps"] = 7;
    add_move_angle_adaptation(settings, 0.1, 2, 0.5, 1.0e-6, 0.1, 3);
  });
  const auto initial_sigma = move_angle_sigma(*solver);
  for (auto step = 0; step < 3; ++step) {
    solver->run();
  }
  const auto frozen_sigma = move_angle_sigma(*solver);
  EXPECT_NE(frozen_sigma, initial_sigma);
  EXPECT_TRUE(move_angle_adaptation_frozen(*solver));
  for (auto step = 0; step < 12; ++step) {
    solver->run();
    EXPECT_DOUBLE_EQ(move_angle_sigma(*solver), frozen_sigma);
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, UnrestrictedAdaptationGuardsTemperatureChanges) {
  auto solver = make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
  });
  EXPECT_NO_THROW(solver->run());
  set_solver_temperature(*solver, 1.0);
  EXPECT_THROW(solver->run(), std::runtime_error);
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationIntervalIsIndependentOfOutputInterval) {
  auto solver = make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    settings["output_write_steps"] = 7;
    add_move_angle_adaptation(settings, 0.1, 2, 0.5, 1.0e-6, 0.1);
  });
  const auto initial_sigma = move_angle_sigma(*solver);
  solver->run();
  EXPECT_DOUBLE_EQ(move_angle_sigma(*solver), initial_sigma);
  solver->run();
  EXPECT_LT(move_angle_sigma(*solver), initial_sigma);
}

TEST_F(ConstrainedMCSolverConstraintTest, RejectsInvalidAdaptationSettings) {
  struct InvalidCase {
    std::string setting;
    std::function<void(libconfig::Setting&)> configure;
  };
  const std::vector<InvalidCase> invalid_cases = {
      {"target_acceptance", [this](auto& settings) {
         add_move_angle_adaptation(settings, 0.0, 10, 0.5, 1.0e-6, 0.1);
       }},
      {"interval_steps", [this](auto& settings) {
         add_move_angle_adaptation(settings, 0.1, 0, 0.5, 1.0e-6, 0.1);
       }},
      {"gain", [this](auto& settings) {
         add_move_angle_adaptation(settings, 0.1, 10, 0.0, 1.0e-6, 0.1);
       }},
      {"min_sigma", [this](auto& settings) {
         add_move_angle_adaptation(settings, 0.1, 10, 0.5, 0.0, 0.1);
       }},
      {"max_sigma", [this](auto& settings) {
         add_move_angle_adaptation(settings, 0.1, 10, 0.5, 0.1, 0.01);
       }},
      {"move_angle_sigma", [this](auto& settings) {
         settings["move_angle_sigma"] = 0.2;
         add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
       }},
      {"burn_in_steps", [this](auto& settings) {
         add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1, 101);
       }},
  };

  for (const auto& invalid_case : invalid_cases) {
    SCOPED_TRACE(invalid_case.setting);
    try {
      auto solver = make_configured_solver(0.0, invalid_case.configure);
      FAIL() << "expected invalid adaptation setting to fail";
    } catch (const jams::ConfigException& error) {
      EXPECT_NE(
          std::string(error.what()).find(invalid_case.setting),
          std::string::npos);
    }
    reset_constrained_mc_globals();
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationRequiresEveryEnabledSetting) {
  for (const auto& missing : {
           std::string("target_acceptance"), std::string("interval_steps"),
           std::string("gain"), std::string("min_sigma"), std::string("max_sigma")}) {
    SCOPED_TRACE(missing);
    try {
      auto solver = make_configured_solver(
          0.0, [this, &missing](libconfig::Setting& settings) {
            add_move_angle_adaptation(
                settings,
                missing == "target_acceptance" ? std::nullopt
                                                : std::optional<double>(0.1),
                missing == "interval_steps" ? std::nullopt
                                             : std::optional<int>(10),
                missing == "gain" ? std::nullopt
                                   : std::optional<double>(0.5),
                missing == "min_sigma" ? std::nullopt
                                        : std::optional<double>(1.0e-6),
                missing == "max_sigma" ? std::nullopt
                                        : std::optional<double>(0.1));
          });
      FAIL() << "expected missing adaptation setting to fail";
    } catch (const jams::ConfigException& error) {
      EXPECT_NE(std::string(error.what()).find(missing), std::string::npos);
    }
    reset_constrained_mc_globals();
  }
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationRejectsZeroAngleMoveFraction) {
  EXPECT_THROW(
      make_configured_solver(0.0, [this](libconfig::Setting& settings) {
        settings.add("move_fraction_uniform", libconfig::Setting::TypeFloat) = 1.0;
        settings.add("move_fraction_angle", libconfig::Setting::TypeFloat) = 0.0;
        settings.add("move_fraction_reflection", libconfig::Setting::TypeFloat) = 0.0;
        add_move_angle_adaptation(settings, 0.1, 10, 0.5, 1.0e-6, 0.1);
      }),
      jams::ConfigException);
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptationLogsInitialUpdatedBoundedAndFrozenStates) {
  auto solver = make_configured_solver(0.0, [this](libconfig::Setting& settings) {
    add_move_angle_adaptation(settings, 0.9, 10, 100.0, 0.01, 0.1, 10);
  });
  auto output = initialization_output(*solver);
  output += perform_adaptation_update(*solver, 10, 100, 0);
  EXPECT_NE(output.find("move angle adaptation enabled"), std::string::npos);
  EXPECT_NE(output.find("initial sigma"), std::string::npos);
  EXPECT_NE(output.find("attempted 100, accepted 0"), std::string::npos);
  EXPECT_NE(output.find("bound minimum"), std::string::npos);
  EXPECT_NE(output.find("frozen production sigma"), std::string::npos);
}

TEST_F(ConstrainedMCSolverConstraintTest, AdaptiveTrajectoriesAreSeedDeterministic) {
  const auto run_case = [this] {
    auto solver = make_configured_solver(300.0, [this](libconfig::Setting& settings) {
      settings["output_write_steps"] = 7;
      add_move_angle_adaptation(settings, 0.1, 2, 0.5, 1.0e-6, 0.1, 8);
    });
    jams::instance().random_generator().seed(445566u);
    std::vector<double> sigmas;
    for (auto step = 0; step < 12; ++step) {
      solver->run();
      sigmas.push_back(move_angle_sigma(*solver));
    }
    solver.reset();
    reset_constrained_mc_globals();
    return sigmas;
  };

  EXPECT_EQ(run_case(), run_case());
}

#endif  // JAMS_TEST_SOLVERS_TEST_CPU_MONTE_CARLO_CONSTRAINED_H
