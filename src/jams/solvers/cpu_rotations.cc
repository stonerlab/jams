//
// Created by Joseph Barker on 2019-05-03.
//

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "jams/core/hamiltonian.h"
#include "jams/core/lattice.h"
#include "jams/solvers/cpu_rotations.h"

#include "jams/interface/config.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"

namespace {

bool contains_any_setting(
    const libconfig::Setting& settings,
    const std::initializer_list<const char*> names) {
  for (const auto* name : names) {
    if (settings.exists(name)) {
      return true;
    }
  }

  return false;
}

void require_numeric_setting(const libconfig::Setting& setting, const char* name) {
  if (!setting.isNumber()) {
    throw jams::ConfigException(setting, name, " must be numeric");
  }
}

std::vector<double> make_angle_range_degrees(
    const libconfig::Setting& setting,
    const char* name,
    const double start_deg,
    const double stop_deg,
    const unsigned count,
    const bool endpoint) {
  if (count == 0) {
    throw jams::ConfigException(setting, name, " count must be greater than zero");
  }

  if (count == 1) {
    return {deg_to_rad(start_deg)};
  }

  std::vector<double> values;
  values.reserve(count);

  const double denominator = endpoint ? static_cast<double>(count - 1) : static_cast<double>(count);
  const double step_deg = (stop_deg - start_deg) / denominator;
  for (unsigned i = 0; i < count; ++i) {
    values.push_back(deg_to_rad(start_deg + step_deg * static_cast<double>(i)));
  }

  return values;
}

std::vector<double> read_angle_values_degrees(const libconfig::Setting& setting, const char* name) {
  if (!jams::is_sequence_setting(setting) || setting.getLength() == 0) {
    throw jams::ConfigException(setting, name, " must be a non-empty array or list");
  }

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(setting.getLength()));
  for (auto i = 0; i < setting.getLength(); ++i) {
    require_numeric_setting(setting[i], name);
    values.push_back(deg_to_rad(static_cast<double>(setting[i])));
  }

  return values;
}

std::vector<double> read_angle_spec_degrees(
    const libconfig::Setting& settings,
    const char* name,
    const unsigned legacy_count,
    const double legacy_start_deg,
    const double legacy_stop_deg) {
  if (!settings.exists(name)) {
    return make_angle_range_degrees(settings, name, legacy_start_deg, legacy_stop_deg, legacy_count, true);
  }

  const auto& spec = settings[name];
  if (spec.isNumber()) {
    return {deg_to_rad(static_cast<double>(spec))};
  }

  if (!spec.isGroup()) {
    throw jams::ConfigException(spec, name, " must be a number or group");
  }

  const bool has_value = spec.exists("value_deg");
  const bool has_values = spec.exists("values_deg");
  const bool has_range = contains_any_setting(spec, {"start_deg", "stop_deg", "count", "endpoint"});

  if ((has_value ? 1 : 0) + (has_values ? 1 : 0) + (has_range ? 1 : 0) > 1) {
    throw jams::ConfigException(spec, name, " must specify only one of value_deg, values_deg or a range");
  }

  if (has_value) {
    const auto& value = spec["value_deg"];
    require_numeric_setting(value, "value_deg");
    return {deg_to_rad(static_cast<double>(value))};
  }

  if (has_values) {
    return read_angle_values_degrees(spec["values_deg"], "values_deg");
  }

  const auto count = jams::config_optional<unsigned>(spec, "count", legacy_count);
  const auto start_deg = jams::config_optional<double>(spec, "start_deg", legacy_start_deg);
  const auto stop_deg = jams::config_optional<double>(spec, "stop_deg", legacy_stop_deg);
  const auto endpoint = jams::config_optional<bool>(spec, "endpoint", true);
  return make_angle_range_degrees(spec, name, start_deg, stop_deg, count, endpoint);
}

}  // namespace

void RotationSolver::initialize(const libconfig::Setting& settings) {
  rotate_all_spins_= jams::config_optional<bool>(settings, "rotate_all_spins", rotate_all_spins_);
  num_theta_ = jams::config_optional<unsigned>(settings, "num_theta", num_theta_);
  num_phi_ = jams::config_optional<unsigned>(settings, "num_phi", num_phi_);

  if (num_theta_ == 0 || num_phi_ == 0) {
    throw std::runtime_error("rotations-cpu requires num_theta and num_phi to be greater than zero");
  }

  theta_values_ = read_angle_spec_degrees(settings, "theta", num_theta_, 0.0, 180.0);
  phi_values_ = read_angle_spec_degrees(settings, "phi", num_phi_, 0.0, 360.0);

  const auto num_spin_targets = rotate_all_spins_ ? 1u : static_cast<unsigned>(globals::lattice->num_basis_sites());
  max_steps_ = static_cast<int>(num_spin_targets * rotation_grid_size());
}

void RotationSolver::run() {
  prepare_rotation_run();

  ++iteration_;
  if (is_running()) {
    apply_current_rotation();
  } else if (!rotate_all_spins_) {
    globals::s = initial_spins_;
  }

  time_ = 0.0;
}

bool RotationSolver::is_running() {
  prepare_rotation_run();
  return Solver::is_running();
}

std::vector<jams::output::ColDef> RotationSolver::monitor_coordinate_columns() const {
  std::vector<jams::output::ColDef> cols;
  cols.push_back({"rotation_step", "steps", jams::output::ColFmt::Integer});

  if (!rotate_all_spins_) {
    cols.push_back({"spin_index", "index", jams::output::ColFmt::Integer});
  }

  cols.push_back({"phi_deg", "degrees", jams::output::ColFmt::Fixed});
  cols.push_back({"theta_deg", "degrees", jams::output::ColFmt::Fixed});
  return cols;
}

void RotationSolver::append_monitor_coordinates(std::vector<double>& values) const {
  values.push_back(iteration());
  if (!rotate_all_spins_) {
    values.push_back(current_spin_index());
  }
  values.push_back(rad_to_deg(current_phi()));
  values.push_back(rad_to_deg(current_theta()));
}

void RotationSolver::prepare_rotation_run() {
  if (prepared_) {
    return;
  }

  initial_spins_ = globals::s;
  prepared_ = true;

  if (Solver::is_running()) {
    apply_current_rotation();
  }
}

void RotationSolver::apply_current_rotation() {
  globals::s = initial_spins_;

  const double theta = current_theta();
  const double phi = current_phi();

  if (rotate_all_spins_) {
    const jams::Mat<double, 3, 3> rotation = rotation_matrix_z(-phi) * rotation_matrix_y(-theta);

    for (auto i = 0; i < globals::num_spins; ++i) {
      jams::Vec<double, 3> spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
      spin = rotation * spin;
      for (auto j = 0; j < 3; ++j) {
        globals::s(i, j) = spin[j];
      }
    }
    return;
  }

  const auto spin_index = current_spin_index();
  const jams::Vec<double, 3> spin = jams::spherical_to_cartesian_vector(1.0, theta, phi);
  for (auto j = 0; j < 3; ++j) {
    globals::s(spin_index, j) = spin[j];
  }
}

unsigned RotationSolver::current_spin_index() const {
  if (rotate_all_spins_) {
    return 0;
  }

  return static_cast<unsigned>(iteration_) / rotation_grid_size();
}

unsigned RotationSolver::current_theta_index() const {
  auto rotation_step = static_cast<unsigned>(iteration_);
  if (!rotate_all_spins_) {
    rotation_step %= rotation_grid_size();
  }

  return rotation_step / static_cast<unsigned>(phi_values_.size());
}

unsigned RotationSolver::current_phi_index() const {
  auto rotation_step = static_cast<unsigned>(iteration_);
  if (!rotate_all_spins_) {
    rotation_step %= rotation_grid_size();
  }

  return rotation_step % static_cast<unsigned>(phi_values_.size());
}

double RotationSolver::current_theta() const {
  return theta_values_[current_theta_index()];
}

double RotationSolver::current_phi() const {
  return phi_values_[current_phi_index()];
}

unsigned RotationSolver::rotation_grid_size() const {
  return static_cast<unsigned>(theta_values_.size() * phi_values_.size());
}
