//
// Created by Joseph Barker on 2019-05-03.
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
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

void append_string_selector_values(
    const libconfig::Setting& settings,
    const char* name,
    std::vector<std::string>& values) {
  if (!settings.exists(name)) {
    return;
  }

  const auto& setting = settings[name];
  if (setting.isString()) {
    values.push_back(jams::read_string_setting(setting, name));
    return;
  }

  if (!jams::is_sequence_setting(setting) || setting.getLength() == 0) {
    throw jams::ConfigException(setting, name, " must be a string or non-empty string array");
  }

  for (auto i = 0; i < setting.getLength(); ++i) {
    values.push_back(jams::read_string_setting(setting[i], name));
  }
}

void append_position_selector_values(
    const libconfig::Setting& settings,
    const char* name,
    std::vector<int>& zero_indexed_values) {
  if (!settings.exists(name)) {
    return;
  }

  const auto& setting = settings[name];
  if (jams::is_integer_setting(setting)) {
    zero_indexed_values.push_back(jams::read_integer_setting(setting, name) - 1);
    return;
  }

  if (!jams::is_sequence_setting(setting) || setting.getLength() == 0) {
    throw jams::ConfigException(setting, name, " must be an integer or non-empty integer array");
  }

  for (auto i = 0; i < setting.getLength(); ++i) {
    zero_indexed_values.push_back(jams::read_integer_setting(setting[i], name) - 1);
  }
}

}  // namespace

void RotationSolver::initialize(const libconfig::Setting& settings) {
  rotate_all_spins_= jams::config_optional<bool>(settings, "rotate_all_spins", rotate_all_spins_);
  num_theta_ = jams::config_optional<unsigned>(settings, "num_theta", num_theta_);
  num_phi_ = jams::config_optional<unsigned>(settings, "num_phi", num_phi_);

  if (num_theta_ == 0 || num_phi_ == 0) {
    throw std::runtime_error("rotations-cpu requires num_theta and num_phi to be greater than zero");
  }

  if (settings.exists("rotation_targets")) {
    if (settings.exists("rotate_all_spins")) {
      throw jams::ConfigException(settings, "rotation_targets", " cannot be combined with rotate_all_spins");
    }
    if (settings.exists("theta") || settings.exists("phi")) {
      throw jams::ConfigException(
          settings,
          "rotation_targets",
          " cannot be combined with top-level theta or phi; put angle settings inside each target");
    }

    read_rotation_targets(settings["rotation_targets"]);

    std::uint64_t total_grid_size = 1;
    for (std::size_t target_index = 0; target_index < rotation_targets_.size(); ++target_index) {
      total_grid_size *= target_grid_size(target_index);
      if (total_grid_size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("rotations-cpu rotation_targets grid is too large");
      }
    }

    max_steps_ = static_cast<int>(total_grid_size);
    return;
  }

  theta_values_ = read_angle_spec_degrees(settings, "theta", num_theta_, 0.0, 180.0);
  phi_values_ = read_angle_spec_degrees(settings, "phi", num_phi_, 0.0, 360.0);

  const auto num_spin_targets = rotate_all_spins_ ? 1u : static_cast<unsigned>(globals::lattice->num_basis_sites());
  max_steps_ = static_cast<int>(num_spin_targets * rotation_grid_size());
}

void RotationSolver::read_rotation_targets(const libconfig::Setting& settings) {
  if (!jams::is_sequence_setting(settings) || settings.getLength() == 0) {
    throw jams::ConfigException(settings, "rotation_targets", " must be a non-empty list");
  }

  if (globals::lattice == nullptr) {
    throw std::runtime_error("rotations-cpu rotation_targets require an initialised lattice");
  }

  std::set<std::string> target_names;
  std::vector<int> spin_owner(static_cast<std::size_t>(globals::num_spins), -1);

  rotation_targets_.clear();
  rotation_targets_.reserve(static_cast<std::size_t>(settings.getLength()));

  for (auto target_index = 0; target_index < settings.getLength(); ++target_index) {
    const auto& target_setting = settings[target_index];
    if (!target_setting.isGroup()) {
      throw jams::ConfigException(target_setting, "rotation target", " must be a group");
    }

    RotationTarget target;
    target.name = jams::config_required<std::string>(target_setting, "name");
    if (target.name.empty()) {
      throw jams::ConfigException(target_setting, "name", " must not be empty");
    }
    if (!target_names.insert(target.name).second) {
      throw jams::ConfigException(target_setting, "name", " must be unique");
    }

    std::vector<std::string> material_names;
    append_string_selector_values(target_setting, "material", material_names);
    append_string_selector_values(target_setting, "materials", material_names);

    std::vector<int> material_indices;
    material_indices.reserve(material_names.size());
    for (const auto& material_name : material_names) {
      if (!globals::lattice->material_exists(material_name)) {
        throw jams::ConfigException(target_setting, "material", " is not defined: ", material_name);
      }
      material_indices.push_back(globals::lattice->material_index(material_name));
    }
    std::sort(material_indices.begin(), material_indices.end());
    material_indices.erase(std::unique(material_indices.begin(), material_indices.end()), material_indices.end());

    std::vector<int> position_indices;
    append_position_selector_values(target_setting, "position", position_indices);
    append_position_selector_values(target_setting, "positions", position_indices);
    append_position_selector_values(target_setting, "unit_cell_position", position_indices);
    append_position_selector_values(target_setting, "unit_cell_positions", position_indices);

    for (const auto position_index : position_indices) {
      if (position_index < 0 || position_index >= globals::lattice->num_basis_sites()) {
        throw jams::ConfigException(
            target_setting,
            "position",
            " must be between 1 and ",
            globals::lattice->num_basis_sites());
      }
    }
    std::sort(position_indices.begin(), position_indices.end());
    position_indices.erase(std::unique(position_indices.begin(), position_indices.end()), position_indices.end());

    if (material_indices.empty() && position_indices.empty()) {
      throw jams::ConfigException(
          target_setting,
          "rotation target",
          " must specify at least one material or unit-cell position");
    }

    for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      const auto material_index = globals::lattice->lattice_site_material_id(spin_index);
      const auto position_index = static_cast<int>(globals::lattice->lattice_site_basis_index(spin_index));

      const bool matches_material = std::binary_search(
          material_indices.begin(),
          material_indices.end(),
          material_index);
      const bool matches_position = std::binary_search(
          position_indices.begin(),
          position_indices.end(),
          position_index);

      if (matches_material || matches_position) {
        target.spin_indices.push_back(spin_index);
      }
    }

    if (target.spin_indices.empty()) {
      throw jams::ConfigException(target_setting, "rotation target", " selects no spins");
    }

    target.theta_values = read_angle_spec_degrees(target_setting, "theta", num_theta_, 0.0, 180.0);
    target.phi_values = read_angle_spec_degrees(target_setting, "phi", num_phi_, 0.0, 360.0);

    for (const auto spin_index : target.spin_indices) {
      if (spin_owner[static_cast<std::size_t>(spin_index)] >= 0) {
        throw jams::ConfigException(
            target_setting,
            "rotation target",
            " overlaps with an earlier target");
      }
      spin_owner[static_cast<std::size_t>(spin_index)] = target_index;
    }

    rotation_targets_.push_back(std::move(target));
  }
}

void RotationSolver::run() {
  prepare_rotation_run();

  ++iteration_;
  if (is_running()) {
    apply_current_rotation();
  } else if (!rotate_all_spins_ || has_rotation_targets()) {
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

  if (has_rotation_targets()) {
    for (const auto& target : rotation_targets_) {
      cols.push_back({"theta_" + target.name + "_deg", "degrees", jams::output::ColFmt::Fixed});
      cols.push_back({"phi_" + target.name + "_deg", "degrees", jams::output::ColFmt::Fixed});
    }
    return cols;
  }

  if (!rotate_all_spins_) {
    cols.push_back({"spin_index", "index", jams::output::ColFmt::Integer});
  }

  cols.push_back({"phi_deg", "degrees", jams::output::ColFmt::Fixed});
  cols.push_back({"theta_deg", "degrees", jams::output::ColFmt::Fixed});
  return cols;
}

void RotationSolver::append_monitor_coordinates(std::vector<double>& values) const {
  values.push_back(iteration());

  if (has_rotation_targets()) {
    for (std::size_t target_index = 0; target_index < rotation_targets_.size(); ++target_index) {
      values.push_back(rad_to_deg(current_target_theta(target_index)));
      values.push_back(rad_to_deg(current_target_phi(target_index)));
    }
    return;
  }

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

  if (has_rotation_targets()) {
    for (std::size_t target_index = 0; target_index < rotation_targets_.size(); ++target_index) {
      const auto& target = rotation_targets_[target_index];
      const auto spin = jams::spherical_to_cartesian_vector(
          1.0,
          current_target_theta(target_index),
          current_target_phi(target_index));

      for (const auto spin_index : target.spin_indices) {
        for (auto j = 0; j < 3; ++j) {
          globals::s(spin_index, j) = spin[j];
        }
      }
    }
    return;
  }

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

bool RotationSolver::has_rotation_targets() const {
  return !rotation_targets_.empty();
}

unsigned RotationSolver::target_grid_size(const std::size_t target_index) const {
  const auto& target = rotation_targets_[target_index];
  return static_cast<unsigned>(target.theta_values.size() * target.phi_values.size());
}

unsigned RotationSolver::current_target_grid_index(const std::size_t target_index) const {
  auto rotation_step = static_cast<unsigned>(iteration_);

  for (std::size_t reverse_index = 0; reverse_index < rotation_targets_.size(); ++reverse_index) {
    const auto index = rotation_targets_.size() - reverse_index - 1;
    const auto grid_size = target_grid_size(index);
    const auto grid_index = rotation_step % grid_size;

    if (index == target_index) {
      return grid_index;
    }

    rotation_step /= grid_size;
  }

  return 0;
}

unsigned RotationSolver::current_target_theta_index(const std::size_t target_index) const {
  return current_target_grid_index(target_index)
      / static_cast<unsigned>(rotation_targets_[target_index].phi_values.size());
}

unsigned RotationSolver::current_target_phi_index(const std::size_t target_index) const {
  return current_target_grid_index(target_index)
      % static_cast<unsigned>(rotation_targets_[target_index].phi_values.size());
}

double RotationSolver::current_target_theta(const std::size_t target_index) const {
  return rotation_targets_[target_index].theta_values[current_target_theta_index(target_index)];
}

double RotationSolver::current_target_phi(const std::size_t target_index) const {
  return rotation_targets_[target_index].phi_values[current_target_phi_index(target_index)];
}
