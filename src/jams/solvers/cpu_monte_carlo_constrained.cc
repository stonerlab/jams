// Copyright 2014 Joseph Barker. All rights reserved.
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

#include <libconfig.h++>

#include "cpu_monte_carlo_constrained.h"

#include <jams/common.h>
#include "jams/core/jams++.h"
#include "jams/helpers/error.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/helpers/montecarlo.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/helpers/montecarlo.h"

namespace {
    constexpr double kSpiralCommensurabilityTolerance = 1.0e-8;

    inline double remap_azimuthal_angle_degrees(double x) {
      // remaps an angle in degrees to the range -180.0 < x <= 180.0
      // this matches the mapping expected for atan2 but avoids the ambiguity
      // of -180.0 == 180.0 which can cause issues for rotation matricies
      return (x = fmod(x + 360.0, 360.0)) > 180.0 ? x - 360.0 : x;
    }

    bool is_finite(const jams::Vec<double, 3>& vector) {
      return std::isfinite(vector[0])
          && std::isfinite(vector[1])
          && std::isfinite(vector[2]);
    }

    jams::Mat<double, 3, 3> rotation_to_z(const jams::Vec<double, 3>& direction) {
      const auto theta = jams::polar_angle(direction);
      const auto phi = jams::azimuthal_angle(direction);
      return rotation_matrix_y(-theta) * rotation_matrix_z(-phi);
    }
}

void ConstrainedMCSolver::initialize(const libconfig::Setting& settings) {
  do_spin_initial_alignment_ = jams::config_optional(settings, "auto_align", do_spin_initial_alignment_);

  const auto constraint_type_name = lowercase(jams::config_optional<std::string>(
      settings, "cmc_constraint_type", "material_transform"));
  if (constraint_type_name == "magnetisation") {
    constraint_type_ = ConstraintType::Magnetisation;
  } else if (constraint_type_name == "material_transform") {
    constraint_type_ = ConstraintType::MaterialTransform;
  } else {
    throw jams::ConfigException(
        settings["cmc_constraint_type"],
        "must be either 'magnetisation' or 'material_transform'");
  }

  const auto constraint_mode_name = lowercase(jams::config_optional<std::string>(
      settings, "cmc_constraint_mode", "global"));
  if (constraint_mode_name == "global") {
    constraint_mode_ = ConstraintMode::Global;
  } else if (constraint_mode_name == "spin_spiral") {
    constraint_mode_ = ConstraintMode::SpinSpiral;
  } else {
    throw jams::ConfigException(
        settings["cmc_constraint_mode"],
        "must be either 'global' or 'spin_spiral'");
  }

  const bool has_spiral_wavevector = settings.exists("cmc_spiral_wavevector");
  const bool has_spiral_axis = settings.exists("cmc_spiral_axis");
  if (constraint_mode_ == ConstraintMode::Global
      && (has_spiral_wavevector || has_spiral_axis)) {
    throw jams::ConfigException(
        settings,
        "cmc_spiral_wavevector and cmc_spiral_axis are only valid when "
        "cmc_constraint_mode is 'spin_spiral'");
  }
  if (constraint_mode_ == ConstraintMode::SpinSpiral
      && (!has_spiral_wavevector || !has_spiral_axis)) {
    throw jams::ConfigException(
        settings,
        "spin_spiral mode requires both cmc_spiral_wavevector and cmc_spiral_axis");
  }

  max_steps_ = jams::config_required<int>(settings, "max_steps");
  min_steps_ = jams::config_optional<int>(settings, "min_steps", jams::defaults::solver_min_steps);

  // theta is angle for z to x-y plane from 0 to 180
  constraint_theta_ = jams::config_required<double>(settings, "cmc_constraint_theta");
  // phi is angle in the x-y plane from 0 to 360
  constraint_phi_ = jams::config_required<double>(settings, "cmc_constraint_phi");
  constraint_phi_ = remap_azimuthal_angle_degrees(constraint_phi_);

  constraint_tolerance_degrees_ = jams::config_optional<double>(
      settings,
      "cmc_constraint_tolerance",
      jams::defaults::solver_monte_carlo_constraint_angular_tolerance_degrees);
  if (!std::isfinite(constraint_tolerance_degrees_)
      || constraint_tolerance_degrees_ <= 0.0
      || constraint_tolerance_degrees_ > 180.0) {
    throw jams::ConfigException(
        settings["cmc_constraint_tolerance"],
        "must be finite and in the range 0 < cmc_constraint_tolerance <= 180 degrees");
  }

  move_angle_sigma_        = jams::config_optional<double>(settings, "move_angle_sigma", jams::defaults::solver_monte_carlo_move_sigma);
  output_write_steps_      = jams::config_optional<int>(settings, "output_write_steps",  jams::defaults::monitor_output_steps);

  constraint_vector_       = jams::spherical_to_cartesian_vector(1.0, deg_to_rad(constraint_theta_), deg_to_rad(constraint_phi_));

  // from cartesian into the constraint space
  rotation_matrix_         = rotation_matrix_y(-deg_to_rad(constraint_theta_))*rotation_matrix_z(-deg_to_rad(constraint_phi_));
  // from the constraint space back to cartesian
  inverse_rotation_matrix_ = transpose(rotation_matrix_);


  if (settings.exists("move_fraction_uniform") || settings.exists("move_fraction_angle") || settings.exists("move_fraction_reflection")) {
    move_fraction_uniform_    = jams::config_optional<double>(settings, "move_fraction_uniform", 0.0);
    move_fraction_angle_      = jams::config_optional<double>(settings, "move_fraction_angle", 0.0);
    move_fraction_reflection_ = jams::config_optional<double>(settings, "move_fraction_reflection", 0.0);

    double move_fraction_sum = move_fraction_uniform_ + move_fraction_angle_ + move_fraction_reflection_;

    move_fraction_uniform_     /= move_fraction_sum;
    move_fraction_angle_       /= move_fraction_sum;
    move_fraction_reflection_  /= move_fraction_sum;
  }

  constraint_transformations_.assign(globals::num_spins, kIdentityMat3);
  for (int i = 0; i < globals::num_spins; ++i) {
    if (constraint_type_ == ConstraintType::MaterialTransform) {
      constraint_transformations_[i] = globals::lattice->material(
          globals::lattice->lattice_site_material_id(i)).transform;
    }
  }

  if (constraint_mode_ == ConstraintMode::SpinSpiral) {
    initialize_spin_spiral(settings);
  }

  output_initialization_info(std::cout);

  validate_angles();
  validate_rotation_matricies();
  validate_moves();

  if (do_spin_initial_alignment_) {
    align_spins_to_constraint();
  }
  validate_constraint();
}

void ConstrainedMCSolver::initialize_spin_spiral(const libconfig::Setting& settings) {
  spiral_wavevector_ = jams::read_vec_setting<double, 3>(
      settings["cmc_spiral_wavevector"], "cmc_spiral_wavevector");
  spiral_axis_ = jams::read_vec_setting<double, 3>(
      settings["cmc_spiral_axis"], "cmc_spiral_axis");

  if (!is_finite(spiral_wavevector_)) {
    throw jams::ConfigException(
        settings["cmc_spiral_wavevector"],
        "cmc_spiral_wavevector components must be finite");
  }

  int nonzero_wavevector_components = 0;
  for (auto direction = 0; direction < 3; ++direction) {
    if (spiral_wavevector_[direction] != 0.0) {
      ++nonzero_wavevector_components;
      spiral_propagation_direction_ = direction;
    }
  }
  if (nonzero_wavevector_components != 1) {
    throw jams::ConfigException(
        settings["cmc_spiral_wavevector"],
        "cmc_spiral_wavevector must have exactly one non-zero component");
  }

  if (!is_finite(spiral_axis_)) {
    throw jams::ConfigException(
        settings["cmc_spiral_axis"],
        "cmc_spiral_axis components must be finite");
  }
  const double spiral_axis_norm = jams::norm(spiral_axis_);
  if (!std::isfinite(spiral_axis_norm) || spiral_axis_norm == 0.0) {
    throw jams::ConfigException(
        settings["cmc_spiral_axis"],
        "cmc_spiral_axis must have a finite, non-zero length");
  }
  spiral_axis_ /= spiral_axis_norm;

  const double wavevector_component = spiral_wavevector_[spiral_propagation_direction_];
  if (globals::lattice->is_periodic(spiral_propagation_direction_)) {
    const double supercell_turns = wavevector_component
        * static_cast<double>(globals::lattice->size(spiral_propagation_direction_));
    const double nearest_integer_turns = std::round(supercell_turns);
    if (!std::isfinite(supercell_turns)
        || std::abs(supercell_turns - nearest_integer_turns)
            > kSpiralCommensurabilityTolerance) {
      throw jams::ConfigException(
          settings["cmc_spiral_wavevector"],
          "component ", spiral_propagation_direction_, " with supercell length ",
          globals::lattice->size(spiral_propagation_direction_),
          " gives ", supercell_turns,
          " turns; periodic spin spirals must be commensurate within ",
          kSpiralCommensurabilityTolerance,
          " turns of the nearest integer (", nearest_integer_turns, ")");
    }
  }

  std::map<int, int> plane_indices;
  for (auto spin = 0; spin < globals::num_spins; ++spin) {
    const int coordinate = globals::lattice->cell_offset(spin)[spiral_propagation_direction_];
    plane_indices.emplace(coordinate, 0);
  }

  constraint_planes_.reserve(plane_indices.size());
  int plane_index = 0;
  for (auto& [coordinate, index] : plane_indices) {
    index = plane_index++;

    ConstraintPlane plane;
    plane.coordinate = coordinate;
    const double phase = kTwoPi * wavevector_component * static_cast<double>(coordinate);
    plane.target_direction = rotation_matrix_from_axis_angle(spiral_axis_, phase)
        * constraint_vector_;
    plane.rotation_matrix = rotation_to_z(plane.target_direction);
    plane.inverse_rotation_matrix = transpose(plane.rotation_matrix);
    constraint_planes_.push_back(std::move(plane));
  }

  spin_constraint_group_.assign(globals::num_spins, -1);
  for (auto spin = 0; spin < globals::num_spins; ++spin) {
    const int coordinate = globals::lattice->cell_offset(spin)[spiral_propagation_direction_];
    const int group = plane_indices.at(coordinate);
    spin_constraint_group_[spin] = group;
    constraint_planes_[group].spins.push_back(spin);
    if (globals::inv_mus(spin) != 0.0) {
      constraint_planes_[group].magnetic_spins.push_back(spin);
    }
  }

  for (const auto& plane : constraint_planes_) {
    if (plane.magnetic_spins.size() < 2) {
      throw jams::ConfigException(
          settings,
          "spin-spiral plane normal to lattice direction ",
          spiral_propagation_direction_, " at cell coordinate ", plane.coordinate,
          " contains ", plane.magnetic_spins.size(),
          " non-zero-moment spins; at least two are required");
    }
  }
}

void ConstrainedMCSolver::run() {
  // Chooses nspins random spin pairs from the spin system and attempts a
  // Constrained Monte Carlo move on each pair, accepting for either lower
  // energy or with a Boltzmann thermal weighting.
  std::uniform_real_distribution<> uniform_distribution;

  jams::montecarlo::MonteCarloUniformMove<jams::RandomGeneratorType> uniform_move(&jams::instance().random_generator());
  jams::montecarlo::MonteCarloAngleMove<jams::RandomGeneratorType>   angle_move(&jams::instance().random_generator(), move_angle_sigma_);
  jams::montecarlo::MonteCarloReflectionMove           reflection_move;

  auto uniform_random_number = uniform_distribution(jams::instance().random_generator());
  if (uniform_random_number < move_fraction_uniform_) {
    move_running_acceptance_count_uniform_ += AsselinAlgorithm(uniform_move);
    run_count_uniform_++;
  } else if (uniform_random_number < (move_fraction_uniform_ + move_fraction_angle_)) {
    move_running_acceptance_count_angle_ += AsselinAlgorithm(angle_move);
    run_count_angle_++;
  } else {
    move_running_acceptance_count_reflection_ += AsselinAlgorithm(reflection_move);
    run_count_reflection_++;
  }

  iteration_++;
  time_ = iteration_;

  if (iteration_ % output_write_steps_ == 0) {
    validate_constraint();

    sum_running_acceptance_statistics();
    output_running_stats_info(std::cout);
    reset_running_statistics();
  }
}

std::vector<jams::output::ColDef> ConstrainedMCSolver::monitor_coordinate_columns() const {
  return {{"step", "steps", jams::output::ColFmt::Integer}};
}

void ConstrainedMCSolver::append_monitor_coordinates(std::vector<double>& values) const {
  values.push_back(iteration());
}

unsigned ConstrainedMCSolver::AsselinAlgorithm(const std::function<jams::Vec<double, 3>(jams::Vec<double, 3>)>& trial_spin_move) {
  std::uniform_real_distribution<> uniform_distribution;

  const double temperature = physics_module_->temperature();
  auto order_parameters = constraint_vectors();

  unsigned moves_accepted = 0;

  // we move two spins moving all spins on average is num_spins/2
  for (auto i = 0; i < globals::num_spins/2; ++i) {
    // Randomly get two spins s1 != s2. For a spin spiral both spins must
    // belong to the same constrained plane.
    auto s1 = jams::montecarlo::random_spin_index();
    const int group = constraint_group_index(s1);

    auto s2 = s1;
    if (constraint_mode_ == ConstraintMode::Global) {
      while (s2 == s1) {
        s2 = jams::montecarlo::random_spin_index();
      }
    } else {
      if (globals::inv_mus(s1) == 0.0) {
        continue;
      }
      const auto& compensation_spins = constraint_planes_[group].magnetic_spins;
      while (s2 == s1) {
        const auto partner_index = jams::instance().random_generator()(compensation_spins.size());
        s2 = compensation_spins[partner_index];
      }
    }

    jams::Vec<double, 3> s1_initial         = jams::montecarlo::get_spin(s1);

    jams::Vec<double, 3> s1_initial_rotated = rotate_cartesian_to_constraint(s1, s1_initial);

    jams::Vec<double, 3> s1_trial           = trial_spin_move(s1_initial);
    jams::Vec<double, 3> s1_trial_rotated   = rotate_cartesian_to_constraint(s1, s1_trial);

    jams::Vec<double, 3> s2_initial         = jams::montecarlo::get_spin(s2);
    jams::Vec<double, 3> s2_initial_rotated = rotate_cartesian_to_constraint(s2, s2_initial);

    if (globals::inv_mus(s1) == 0.0 || globals::inv_mus(s2) == 0.0) {
      continue;
    }

    // calculate new spin based on contraint mx = my = 0 in the constraint vector reference frame
    jams::Vec<double, 3> s2_trial_rotated   = s2_initial_rotated + (s1_initial_rotated - s1_trial_rotated ) * (globals::mus(s1) * globals::inv_mus(s2)) ;

    double ss2 = s2_trial_rotated[0] * s2_trial_rotated[0] + s2_trial_rotated[1] * s2_trial_rotated[1];
    if (ss2 > 1.0) {
      // the rotated spin does not fit on the unit sphere - revert s1 and reject move
      continue;
    }
    // calculate the z-component so that |s2| = 1
    s2_trial_rotated[2] = copysign(sqrt(1.0 - ss2), s2_initial_rotated[2]);

    jams::Vec<double, 3> s2_trial = rotate_constraint_to_cartesian(s2, s2_trial_rotated);

    jams::Vec<double, 3> delta_order_parameter = constraint_vector_difference(
        s1, s1_initial, s1_trial, s2, s2_initial, s2_trial);

    const auto& group_rotation_matrix = rotation_matrix(group);
    auto& order_parameter = order_parameters[group];
    jams::Vec<double, 3> trial_order_parameter_rotated =
        group_rotation_matrix * (order_parameter + delta_order_parameter);

    if (trial_order_parameter_rotated[2] < 0.0) {
      // The new order parameter is in the opposite sense - reject the move.
      continue;
    }

    jams::Vec<double, 3> initial_order_parameter_rotated =
        group_rotation_matrix * order_parameter;

    // calculate the Boltzmann weighted probability including the Jacobian factors (see paper)
    double delta_e = energy_difference(s1, s1_initial, s1_trial, s2, s2_initial, s2_trial);

    if (temperature == 0.0) {
      // At zero temperature, only strictly energy-lowering moves are accepted.
      if (delta_e >= 0.0) {
        continue;
      }
    } else {
      const double beta = 1.0 / (temperature * kBoltzmannIU);
      double jacobian_factor = pow2(trial_order_parameter_rotated[2] / initial_order_parameter_rotated[2])
          * abs(s2_initial_rotated[2] / s2_trial_rotated[2]);
      double probability = std::min(1.0, exp(-delta_e * beta) * jacobian_factor);

      if (uniform_distribution(jams::instance().random_generator()) > probability) {
        // reject move
        continue;
      }
    }

    // accept move
    jams::montecarlo::set_spin(s1, s1_trial);
    jams::montecarlo::set_spin(s2, s2_trial);

    order_parameter += delta_order_parameter;

    moves_accepted++;
  }

  return moves_accepted;
}

double ConstrainedMCSolver::energy_difference(const int &s1, const jams::Vec<double, 3> &s1_initial, const jams::Vec<double, 3> &s1_trial,
                                              const int &s2, const jams::Vec<double, 3> &s2_initial, const jams::Vec<double, 3> &s2_trial) const {
  assert(s1 != s2);
  double delta_energy1 = 0.0;
  for (const auto& hamiltonian : hamiltonians_) {
    delta_energy1 += hamiltonian->calculate_energy_difference(s1, s1_initial, s1_trial, this->time());
  }

  // temporarily accept the move for s1 so we can calculate the s2 energies
  jams::montecarlo::set_spin(s1, s1_trial);
  double delta_energy2 = 0.0;
  for (const auto& hamiltonian : hamiltonians_) {
    delta_energy2 += hamiltonian->calculate_energy_difference(s2, s2_initial, s2_trial, this->time());
  }
  jams::montecarlo::set_spin(s1, s1_initial);

  return delta_energy1 + delta_energy2;
}

jams::Vec<double, 3> ConstrainedMCSolver::constraint_vector_difference(const int &s1, const jams::Vec<double, 3> &s1_initial, const jams::Vec<double, 3> &s1_trial,
                                                   const int &s2, const jams::Vec<double, 3> &s2_initial, const jams::Vec<double, 3> &s2_trial) const {
  return globals::mus(s1) * spin_to_order_parameter(s1, s1_trial - s1_initial)
      + globals::mus(s2) * spin_to_order_parameter(s2, s2_trial - s2_initial);
}

jams::Vec<double, 3> ConstrainedMCSolver::total_constraint_vector() const {
  jams::Vec<double, 3> total = {0.0, 0.0, 0.0};

  for (auto i = 0; i < globals::num_spins; ++i) {
    total += globals::mus(i) * spin_to_order_parameter(i, jams::montecarlo::get_spin(i));
  }

  return total;
}

std::vector<jams::Vec<double, 3>> ConstrainedMCSolver::constraint_vectors() const {
  if (constraint_mode_ == ConstraintMode::Global) {
    return {total_constraint_vector()};
  }

  std::vector<jams::Vec<double, 3>> totals(
      constraint_planes_.size(), jams::Vec<double, 3>{{0.0, 0.0, 0.0}});
  for (auto spin = 0; spin < globals::num_spins; ++spin) {
    totals[spin_constraint_group_[spin]] += globals::mus(spin)
        * spin_to_order_parameter(spin, jams::montecarlo::get_spin(spin));
  }
  return totals;
}

int ConstrainedMCSolver::constraint_group_index(const int spin_index) const {
  if (constraint_mode_ == ConstraintMode::Global) {
    return 0;
  }
  return spin_constraint_group_[spin_index];
}

const jams::Vec<double, 3>& ConstrainedMCSolver::target_direction(const int group_index) const {
  if (constraint_mode_ == ConstraintMode::Global) {
    return constraint_vector_;
  }
  return constraint_planes_[group_index].target_direction;
}

const jams::Mat<double, 3, 3>& ConstrainedMCSolver::rotation_matrix(const int group_index) const {
  if (constraint_mode_ == ConstraintMode::Global) {
    return rotation_matrix_;
  }
  return constraint_planes_[group_index].rotation_matrix;
}

const jams::Mat<double, 3, 3>& ConstrainedMCSolver::inverse_rotation_matrix(const int group_index) const {
  if (constraint_mode_ == ConstraintMode::Global) {
    return inverse_rotation_matrix_;
  }
  return constraint_planes_[group_index].inverse_rotation_matrix;
}

jams::Vec<double, 3> ConstrainedMCSolver::spin_to_order_parameter(const int &i, const jams::Vec<double, 3> &spin) const {
  return constraint_transformations_[i] * spin;
}

jams::Vec<double, 3> ConstrainedMCSolver::order_parameter_to_spin(const int &i, const jams::Vec<double, 3> &spin) const {
  return transpose(constraint_transformations_[i]) * spin;
}

jams::Vec<double, 3> ConstrainedMCSolver::rotate_cartesian_to_constraint(const int &i, const jams::Vec<double, 3> &spin) const {
  return rotation_matrix(constraint_group_index(i)) * spin_to_order_parameter(i, spin);
}

jams::Vec<double, 3> ConstrainedMCSolver::rotate_constraint_to_cartesian(const int &i, const jams::Vec<double, 3> &spin) const {
  return order_parameter_to_spin(
      i, inverse_rotation_matrix(constraint_group_index(i)) * spin);
}

const char* ConstrainedMCSolver::constraint_type_name() const {
  switch (constraint_type_) {
    case ConstraintType::Magnetisation:
      return "magnetisation";
    case ConstraintType::MaterialTransform:
      return "material_transform";
  }
  return "unknown";
}

const char* ConstrainedMCSolver::constraint_mode_name() const {
  switch (constraint_mode_) {
    case ConstraintMode::Global:
      return "global";
    case ConstraintMode::SpinSpiral:
      return "spin_spiral";
  }
  return "unknown";
}

void ConstrainedMCSolver::output_initialization_info(std::ostream &os) {
  os << "    constraint mode " << constraint_mode_name() << "\n";
  os << "    constraint type " << constraint_type_name() << "\n";
  os << "    constraint angle theta (deg) " << constraint_theta_ << "\n";
  os << "    constraint angle phi (deg) " << constraint_phi_ << "\n";
  os << "    constraint angular tolerance (deg) " << constraint_tolerance_degrees_ << "\n";
  os << "    "
     << (constraint_mode_ == ConstraintMode::Global
             ? "constraint vector " : "reference constraint vector ")
     << constraint_vector_[0] << " " << constraint_vector_[1] << " "
     << constraint_vector_[2] << "\n";
  if (constraint_mode_ == ConstraintMode::SpinSpiral) {
    os << "    spiral wavevector " << spiral_wavevector_ << " (cycles per unit cell)\n";
    os << "    spiral rotation axis " << spiral_axis_ << "\n";
    os << "    spiral propagation lattice direction " << spiral_propagation_direction_ << "\n";
    os << "    constrained planes " << constraint_planes_.size() << "\n";
    for (const auto& plane : constraint_planes_) {
      os << "      coordinate " << plane.coordinate
         << ": " << plane.magnetic_spins.size() << " magnetic spins\n";
    }
  }
  os << "    move_fraction_uniform " << move_fraction_uniform_ << "\n";
  os << "    move_fraction_angle " << move_fraction_angle_ << "\n";
  os << "    move_fraction_reflection " << move_fraction_reflection_ << "\n";
  os << "    move_angle_sigma " << move_angle_sigma_ << "\n";
  os << "    output_write_steps " << output_write_steps_ << "\n";
  if (constraint_mode_ == ConstraintMode::Global) {
    os << "    rotation matrix m -> mz\n";
    for (auto i = 0; i < 3; ++i) {
      os << "      ";
      for (auto j = 0; j < 3; ++j) {
        os << rotation_matrix_[i][j] << " ";
      }
      os << "\n";
    }
    os << "    inverse rotation matrix mz -> m\n";
    for (auto i = 0; i < 3; ++i) {
      os << "      ";
      for (auto j = 0; j < 3; ++j) {
        os << inverse_rotation_matrix_[i][j] << " ";
      }
      os << "\n";
    }
  }
}

void ConstrainedMCSolver::validate_rotation_matricies() const {
  const jams::Vec<double, 3> constraint_axis = {0.0, 0.0, 1.0};
  const int group_count = constraint_mode_ == ConstraintMode::Global
      ? 1 : static_cast<int>(constraint_planes_.size());

  std::cout << "  rotation sanity check for " << group_count
            << " constraint group" << (group_count == 1 ? "" : "s") << "\n";

  for (auto group = 0; group < group_count; ++group) {
    const auto& requested_direction = target_direction(group);
    const auto test_forward_vec = rotation_matrix(group) * requested_direction;
    const auto test_back_vec = inverse_rotation_matrix(group) * test_forward_vec;

    for (int component = 0; component < 3; ++component) {
      if (!approximately_equal(
              constraint_axis[component], test_forward_vec[component],
              jams::defaults::solver_monte_carlo_constraint_tolerance)
          || !approximately_equal(
              requested_direction[component], test_back_vec[component],
              jams::defaults::solver_monte_carlo_constraint_tolerance)) {
        throw std::runtime_error(
            "ConstrainedMCSolver :: rotation sanity check failed for constraint group "
            + std::to_string(group));
      }
    }
  }
}

void ConstrainedMCSolver::output_running_stats_info(std::ostream &os) {
  os << "\n";
  os << "iteration: " << iteration_ << "\n";
  os << "move_acceptance_fraction:\n";

  double half_num_spins = 0.5 * globals::num_spins;

  os << "  uniform:    ";
  os << division_or_zero(move_running_acceptance_count_uniform_, half_num_spins * run_count_uniform_) << " (";
  os << division_or_zero(move_total_acceptance_count_uniform_,   half_num_spins * move_total_count_uniform_) << ") \n";

  os << "  angle:      ";
  os << division_or_zero(move_running_acceptance_count_angle_, half_num_spins * run_count_angle_) << " (";
  os << division_or_zero(move_total_acceptance_count_angle_,   half_num_spins * move_total_count_angle_) << ") \n";

  os << "  reflection: ";
  os << division_or_zero(move_running_acceptance_count_reflection_, half_num_spins * run_count_reflection_) << " (";
  os << division_or_zero(move_total_acceptance_count_reflection_,   half_num_spins * move_total_count_reflection_) << ") \n";
}


void ConstrainedMCSolver::validate_constraint() const {
  const auto order_parameters = constraint_vectors();
  if (constraint_mode_ == ConstraintMode::Global) {
    validate_constraint_vector(
        order_parameters.front(), constraint_vector_, "the total constraint vector");
    return;
  }

  for (auto group = 0; group < constraint_planes_.size(); ++group) {
    const auto& plane = constraint_planes_[group];
    validate_constraint_vector(
        order_parameters[group],
        plane.target_direction,
        "spin-spiral plane normal to lattice direction "
            + std::to_string(spiral_propagation_direction_)
            + " at cell coordinate " + std::to_string(plane.coordinate));
  }
}

void ConstrainedMCSolver::validate_constraint_vector(
    const jams::Vec<double, 3>& order_parameter,
    const jams::Vec<double, 3>& target_direction,
    const std::string& description) const {
  const double order_parameter_norm = jams::norm(order_parameter);

  if (!is_finite(order_parameter)
      || !std::isfinite(order_parameter_norm)
      || order_parameter_norm == 0.0) {
    std::stringstream ss;
    ss << "ConstrainedMCSolver -- constraint direction is undefined for "
       << description << " because its vector is zero or non-finite ("
       << std::scientific << std::setprecision(12)
       << order_parameter[0] << ", " << order_parameter[1] << ", " << order_parameter[2] << ")";
    throw std::runtime_error(ss.str());
  }

  const double angular_error_degrees = rad_to_deg(std::atan2(
      jams::norm(jams::cross(order_parameter, target_direction)),
      jams::dot(order_parameter, target_direction)));
  if (!std::isfinite(angular_error_degrees)) {
    throw std::runtime_error(
        "ConstrainedMCSolver -- constraint direction is undefined for " + description
        + " because its angular difference is non-finite");
  }

  if (angular_error_degrees > constraint_tolerance_degrees_) {
    const double actual_theta = rad_to_deg(jams::polar_angle(order_parameter));
    const double actual_phi = rad_to_deg(jams::azimuthal_angle(order_parameter));
    const double requested_theta = rad_to_deg(jams::polar_angle(target_direction));
    const double requested_phi = rad_to_deg(jams::azimuthal_angle(target_direction));
    std::stringstream ss;
    ss << "ConstrainedMCSolver -- constraint direction violated for " << description
       << " (requested theta "
       << std::fixed << std::setprecision(10) << requested_theta << " deg, phi "
       << requested_phi << " deg; actual theta " << actual_theta << " deg, phi "
       << actual_phi << " deg; angular difference "
       << std::scientific << std::setprecision(12) << angular_error_degrees
       << " deg exceeds tolerance " << constraint_tolerance_degrees_ << " deg)";
    throw std::runtime_error(ss.str());
  }
}

void ConstrainedMCSolver::validate_angles() const {
  if (!std::isfinite(constraint_theta_) || constraint_theta_ < 0 || constraint_theta_ > 180.0) {
    throw std::runtime_error(
        "ConstrainedMCSolver -- theta ( " + std::to_string(constraint_theta_) + " ) is out of range (0 <= theta <= 180)");
  }

  if (!std::isfinite(constraint_phi_) || constraint_phi_ <= -180.0 || constraint_phi_ > 180.0) {
    throw std::runtime_error(
        "ConstrainedMCSolver -- phi ( " + std::to_string(constraint_phi_) + " ) is out of range (-180 <= phi <= 180)");
  }
}

void ConstrainedMCSolver::validate_moves() const {
  if (approximately_equal(move_fraction_reflection_, 1.0, DBL_EPSILON)) {
    throw std::runtime_error("ConstrainedMCSolver -- Only reflection moves have been configured. This breaks ergodicity.");
  }
}

void ConstrainedMCSolver::reset_running_statistics() {
  move_running_acceptance_count_uniform_    = 0;
  move_running_acceptance_count_angle_      = 0;
  move_running_acceptance_count_reflection_ = 0;

  run_count_uniform_    = 0;
  run_count_angle_      = 0;
  run_count_reflection_ = 0;
}

void ConstrainedMCSolver::sum_running_acceptance_statistics() {
  move_total_count_uniform_    += run_count_uniform_;
  move_total_count_angle_      += run_count_angle_;
  move_total_count_reflection_ += run_count_reflection_;

  move_total_acceptance_count_uniform_    += move_running_acceptance_count_uniform_;
  move_total_acceptance_count_angle_      += move_running_acceptance_count_angle_;
  move_total_acceptance_count_reflection_ += move_running_acceptance_count_reflection_;
}

void ConstrainedMCSolver::align_spins_to_constraint() const {
  const auto require_defined_direction = [](
      const jams::Vec<double, 3>& vector,
      const std::string& description) {
    const double vector_norm = jams::norm(vector);
    if (!is_finite(vector) || !std::isfinite(vector_norm) || vector_norm == 0.0) {
      std::stringstream ss;
      ss << "ConstrainedMCSolver -- constraint direction is undefined for "
         << description << " because its vector is zero or non-finite ("
         << std::scientific << std::setprecision(12)
         << vector[0] << ", " << vector[1] << ", " << vector[2] << ")";
      throw std::runtime_error(ss.str());
    }
  };

  if (constraint_mode_ == ConstraintMode::Global) {
    const auto order_parameter = total_constraint_vector();
    require_defined_direction(order_parameter, "the total constraint vector");
    const auto rotation = rotation_matrix_between_vectors(
        order_parameter, constraint_vector_);

    for (auto spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      const auto spin = jams::montecarlo::get_spin(spin_index);
      const auto aligned_order_parameter_spin = rotation
          * spin_to_order_parameter(spin_index, spin);
      const auto aligned_spin = order_parameter_to_spin(
          spin_index, aligned_order_parameter_spin);
      for (auto component : {0, 1, 2}) {
        globals::s(spin_index, component) = aligned_spin[component];
      }
    }
    return;
  }

  const auto order_parameters = constraint_vectors();
  for (auto group = 0; group < constraint_planes_.size(); ++group) {
    const auto& plane = constraint_planes_[group];
    const auto description = "spin-spiral plane normal to lattice direction "
        + std::to_string(spiral_propagation_direction_)
        + " at cell coordinate " + std::to_string(plane.coordinate);
    require_defined_direction(order_parameters[group], description);
    const auto alignment = rotation_matrix_between_vectors(
        order_parameters[group], plane.target_direction);

    for (const auto spin_index : plane.spins) {
      const auto spin = jams::montecarlo::get_spin(spin_index);
      const auto aligned_order_parameter_spin = alignment
          * spin_to_order_parameter(spin_index, spin);
      const auto aligned_spin = order_parameter_to_spin(
          spin_index, aligned_order_parameter_spin);
      for (auto component : {0, 1, 2}) {
        globals::s(spin_index, component) = aligned_spin[component];
      }
    }
  }
}
