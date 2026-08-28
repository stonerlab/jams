// Copyright 2014 Joseph Barker. All rights reserved.
#include <cmath>
#include <iomanip>
#include <limits>
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
    constexpr double kSpiralOrthogonalityTolerance = 1.0e-8;

    inline double remap_azimuthal_angle_degrees(double x) {
      // remaps an angle in degrees to the range -180.0 < x <= 180.0
      // this matches the mapping expected for atan2 but avoids the ambiguity
      // of -180.0 == 180.0 which can cause issues for rotation matricies
      return (x = fmod(x + 360.0, 360.0)) > 180.0 ? x - 360.0 : x;
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
  const bool has_spiral_profile = settings.exists("cmc_spiral_profile");
  const bool has_spiral_background = settings.exists("cmc_spiral_background");
  const bool has_spiral_polarisation = settings.exists("cmc_spiral_polarisation");
  const bool has_spiral_amplitude = settings.exists("cmc_spiral_amplitude");
  const bool has_spiral_phase = settings.exists("cmc_spiral_phase");
  const bool has_spiral_propagation_direction =
      settings.exists("cmc_spiral_propagation_direction");
  if (constraint_mode_ == ConstraintMode::Global
      && (has_spiral_wavevector || has_spiral_profile
          || has_spiral_background || has_spiral_polarisation
          || has_spiral_amplitude || has_spiral_phase
          || has_spiral_propagation_direction
          || settings.exists("cmc_spiral_axis"))) {
    throw jams::ConfigException(
        settings,
        "cmc_spiral_* settings are only valid when "
        "cmc_constraint_mode is 'spin_spiral'");
  }
  if (constraint_mode_ == ConstraintMode::SpinSpiral) {
    if (settings.exists("cmc_constraint_theta")
        || settings.exists("cmc_constraint_phi")
        || settings.exists("cmc_spiral_axis")) {
      throw jams::ConfigException(
          settings,
          "spin_spiral mode does not accept cmc_constraint_theta, "
          "cmc_constraint_phi, or cmc_spiral_axis");
    }
    if (!has_spiral_wavevector || !has_spiral_profile
        || !has_spiral_background || !has_spiral_polarisation
        || !has_spiral_amplitude) {
      throw jams::ConfigException(
          settings,
          "spin_spiral mode requires cmc_spiral_wavevector, "
          "cmc_spiral_profile, cmc_spiral_background, "
          "cmc_spiral_polarisation, and cmc_spiral_amplitude");
    }
  }

  max_steps_ = jams::config_required<int>(settings, "max_steps");
  min_steps_ = jams::config_optional<int>(settings, "min_steps", jams::defaults::solver_min_steps);

  if (constraint_mode_ == ConstraintMode::Global) {
    // theta is angle from the z axis and phi is the xy-plane azimuth.
    constraint_theta_ = jams::config_required<double>(settings, "cmc_constraint_theta");
    constraint_phi_ = jams::config_required<double>(settings, "cmc_constraint_phi");
    constraint_phi_ = remap_azimuthal_angle_degrees(constraint_phi_);
    constraint_vector_ = jams::spherical_to_cartesian_vector(
        1.0, deg_to_rad(constraint_theta_), deg_to_rad(constraint_phi_));

    // From cartesian into constraint space and back again.
    rotation_matrix_ = rotation_matrix_y(-deg_to_rad(constraint_theta_))
        * rotation_matrix_z(-deg_to_rad(constraint_phi_));
    inverse_rotation_matrix_ = transpose(rotation_matrix_);
  }

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

  if (settings.exists("move_fraction_uniform") || settings.exists("move_fraction_angle") || settings.exists("move_fraction_reflection")) {
    move_fraction_uniform_    = jams::config_optional<double>(settings, "move_fraction_uniform", 0.0);
    move_fraction_angle_      = jams::config_optional<double>(settings, "move_fraction_angle", 0.0);
    move_fraction_reflection_ = jams::config_optional<double>(settings, "move_fraction_reflection", 0.0);

    double move_fraction_sum = move_fraction_uniform_ + move_fraction_angle_ + move_fraction_reflection_;

    move_fraction_uniform_     /= move_fraction_sum;
    move_fraction_angle_       /= move_fraction_sum;
    move_fraction_reflection_  /= move_fraction_sum;
  }

  initialize_move_angle_adaptation(settings);

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

  if (constraint_mode_ == ConstraintMode::Global) {
    validate_angles();
  }
  validate_rotation_matricies();
  validate_moves();

  if (do_spin_initial_alignment_) {
    align_spins_to_constraint();
  }
  validate_constraint();
}

void ConstrainedMCSolver::initialize_move_angle_adaptation(
    const libconfig::Setting& settings) {
  if (!settings.exists("move_angle_adaptation")) {
    return;
  }

  const auto& adaptation = settings["move_angle_adaptation"];
  if (!adaptation.isGroup()) {
    throw jams::ConfigException(
        adaptation, "move_angle_adaptation must be a group");
  }

  if (adaptation.exists("enabled")) {
    if (adaptation["enabled"].getType() != libconfig::Setting::TypeBoolean) {
      throw jams::ConfigException(
          adaptation["enabled"], "move_angle_adaptation.enabled must be boolean");
    }
    move_angle_adaptation_enabled_ = bool(adaptation["enabled"]);
  }
  if (!move_angle_adaptation_enabled_) {
    return;
  }

  jams::require_settings(
      adaptation,
      {"target_acceptance", "interval_steps", "gain", "min_sigma", "max_sigma"},
      " is required when move_angle_adaptation.enabled = true");

  move_angle_target_acceptance_ = jams::read_numeric_setting<double>(
      adaptation["target_acceptance"], "move_angle_adaptation.target_acceptance");
  if (!std::isfinite(move_angle_target_acceptance_)
      || move_angle_target_acceptance_ <= 0.0
      || move_angle_target_acceptance_ >= 1.0) {
    throw jams::ConfigException(
        adaptation["target_acceptance"],
        "move_angle_adaptation.target_acceptance must be finite and in the range 0 < target_acceptance < 1");
  }

  move_angle_adaptation_interval_steps_ = jams::read_integer_in_range(
      adaptation["interval_steps"], "move_angle_adaptation.interval_steps",
      1, std::numeric_limits<int>::max());

  move_angle_adaptation_gain_ = jams::read_numeric_setting<double>(
      adaptation["gain"], "move_angle_adaptation.gain");
  if (!std::isfinite(move_angle_adaptation_gain_)
      || move_angle_adaptation_gain_ <= 0.0) {
    throw jams::ConfigException(
        adaptation["gain"],
        "move_angle_adaptation.gain must be finite and positive");
  }

  move_angle_min_sigma_ = jams::read_numeric_setting<double>(
      adaptation["min_sigma"], "move_angle_adaptation.min_sigma");
  if (!std::isfinite(move_angle_min_sigma_) || move_angle_min_sigma_ <= 0.0) {
    throw jams::ConfigException(
        adaptation["min_sigma"],
        "move_angle_adaptation.min_sigma must be finite and positive");
  }

  move_angle_max_sigma_ = jams::read_numeric_setting<double>(
      adaptation["max_sigma"], "move_angle_adaptation.max_sigma");
  if (!std::isfinite(move_angle_max_sigma_) || move_angle_max_sigma_ <= 0.0) {
    throw jams::ConfigException(
        adaptation["max_sigma"],
        "move_angle_adaptation.max_sigma must be finite and positive");
  }
  if (move_angle_min_sigma_ > move_angle_max_sigma_) {
    throw jams::ConfigException(
        adaptation["max_sigma"],
        "move_angle_adaptation bounds must satisfy min_sigma <= max_sigma");
  }
  if (!std::isfinite(move_angle_sigma_)
      || move_angle_sigma_ < move_angle_min_sigma_
      || move_angle_sigma_ > move_angle_max_sigma_) {
    throw jams::ConfigException(
        settings,
        "move_angle_sigma must be finite and satisfy move_angle_adaptation.min_sigma <= move_angle_sigma <= move_angle_adaptation.max_sigma");
  }

  if (adaptation.exists("burn_in_steps")) {
    const auto burn_in_steps = jams::read_integer_in_range(
        adaptation["burn_in_steps"], "move_angle_adaptation.burn_in_steps",
        1, std::numeric_limits<int>::max());
    if (burn_in_steps > max_steps_) {
      throw jams::ConfigException(
          adaptation["burn_in_steps"],
          "move_angle_adaptation.burn_in_steps must be less than or equal to max_steps (",
          max_steps_, ")");
    }
    move_angle_burn_in_steps_ = burn_in_steps;
  }

  if (!(move_fraction_angle_ > 0.0)) {
    throw jams::ConfigException(
        adaptation,
        "move_angle_adaptation cannot be enabled when move_fraction_angle is zero");
  }

  if (!move_angle_burn_in_steps_.has_value()
      && globals::config != nullptr
      && globals::config->exists("physics.temperature")) {
    const auto configured_temperature = jams::read_numeric_setting<double>(
        globals::config->lookup("physics.temperature"), "physics.temperature");
    if (configured_temperature != 0.0) {
      throw jams::ConfigException(
          adaptation,
          "move_angle_adaptation.burn_in_steps is required when the simulation temperature is nonzero");
    }
  }
}

void ConstrainedMCSolver::initialize_spin_spiral(const libconfig::Setting& settings) {
  spiral_wavevector_ = jams::read_vec_setting<double, 3>(
      settings["cmc_spiral_wavevector"], "cmc_spiral_wavevector");
  spiral_background_ = jams::read_vec_setting<double, 3>(
      settings["cmc_spiral_background"], "cmc_spiral_background");
  spiral_polarisation_ = jams::read_vec_setting<double, 3>(
      settings["cmc_spiral_polarisation"], "cmc_spiral_polarisation");

  const auto profile = lowercase(jams::config_required<std::string>(
      settings, "cmc_spiral_profile"));
  if (profile == "circular") {
    spiral_profile_ = SpinSpiralProfile::Circular;
  } else if (profile == "linear") {
    spiral_profile_ = SpinSpiralProfile::Linear;
  } else {
    throw jams::ConfigException(
        settings["cmc_spiral_profile"],
        "cmc_spiral_profile must be either 'circular' or 'linear'");
  }

  spiral_amplitude_degrees_ = jams::config_required<double>(
      settings, "cmc_spiral_amplitude");
  spiral_phase_degrees_ = jams::config_optional<double>(
      settings, "cmc_spiral_phase", 0.0);
  if (!std::isfinite(spiral_amplitude_degrees_)
      || spiral_amplitude_degrees_ < 0.0
      || (spiral_profile_ == SpinSpiralProfile::Circular
          && spiral_amplitude_degrees_ > 180.0)
      || (spiral_profile_ == SpinSpiralProfile::Linear
          && spiral_amplitude_degrees_ >= 90.0)) {
    throw jams::ConfigException(
        settings["cmc_spiral_amplitude"],
        spiral_profile_ == SpinSpiralProfile::Circular
            ? "circular cmc_spiral_amplitude must be finite and in [0, 180] degrees"
            : "linear cmc_spiral_amplitude must be finite and in [0, 90) degrees");
  }
  if (!std::isfinite(spiral_phase_degrees_)) {
    throw jams::ConfigException(
        settings["cmc_spiral_phase"],
        "cmc_spiral_phase must be finite");
  }
  spiral_phase_degrees_ = remap_azimuthal_angle_degrees(spiral_phase_degrees_);
  spiral_amplitude_ = deg_to_rad(spiral_amplitude_degrees_);
  spiral_phase_ = deg_to_rad(spiral_phase_degrees_);

  if (!jams::is_finite(spiral_wavevector_)) {
    throw jams::ConfigException(
        settings["cmc_spiral_wavevector"],
        "cmc_spiral_wavevector components must be finite");
  }

  int nonzero_wavevector_components = 0;
  int inferred_propagation_direction = -1;
  for (auto direction = 0; direction < 3; ++direction) {
    if (spiral_wavevector_[direction] != 0.0) {
      ++nonzero_wavevector_components;
      inferred_propagation_direction = direction;
    }
  }
  if (nonzero_wavevector_components > 1) {
    throw jams::ConfigException(
        settings["cmc_spiral_wavevector"],
        "cmc_spiral_wavevector must have at most one non-zero component");
  }

  const bool has_explicit_propagation_direction =
      settings.exists("cmc_spiral_propagation_direction");
  int explicit_propagation_direction = -1;
  if (has_explicit_propagation_direction) {
    const auto& direction_setting = settings["cmc_spiral_propagation_direction"];
    explicit_propagation_direction = jams::read_integer_in_range(
        direction_setting, "cmc_spiral_propagation_direction", 0, 2);
  }

  zero_wavevector_plane_constraint_ = nonzero_wavevector_components == 0;
  if (zero_wavevector_plane_constraint_) {
    if (!has_explicit_propagation_direction) {
      throw jams::ConfigException(
          settings["cmc_spiral_wavevector"],
          "a zero cmc_spiral_wavevector requires "
          "cmc_spiral_propagation_direction = 0, 1, or 2");
    }
    spiral_propagation_direction_ = explicit_propagation_direction;
  } else {
    spiral_propagation_direction_ = inferred_propagation_direction;
    if (has_explicit_propagation_direction
        && explicit_propagation_direction != inferred_propagation_direction) {
      throw jams::ConfigException(
          settings["cmc_spiral_propagation_direction"],
          "cmc_spiral_propagation_direction = ", explicit_propagation_direction,
          " conflicts with non-zero cmc_spiral_wavevector component ",
          inferred_propagation_direction);
    }
  }

  if (!jams::is_finite(spiral_background_)) {
    throw jams::ConfigException(
        settings["cmc_spiral_background"],
        "cmc_spiral_background components must be finite");
  }
  const double background_norm = jams::norm(spiral_background_);
  if (!std::isfinite(background_norm) || background_norm == 0.0) {
    throw jams::ConfigException(
        settings["cmc_spiral_background"],
        "cmc_spiral_background must have a finite, non-zero length");
  }
  spiral_background_ /= background_norm;

  if (!jams::is_finite(spiral_polarisation_)) {
    throw jams::ConfigException(
        settings["cmc_spiral_polarisation"],
        "cmc_spiral_polarisation components must be finite");
  }
  const double polarisation_norm = jams::norm(spiral_polarisation_);
  if (!std::isfinite(polarisation_norm) || polarisation_norm == 0.0) {
    throw jams::ConfigException(
        settings["cmc_spiral_polarisation"],
        "cmc_spiral_polarisation must have a finite, non-zero length");
  }
  spiral_polarisation_ /= polarisation_norm;

  const double background_polarisation_dot =
      jams::dot(spiral_background_, spiral_polarisation_);
  if (!std::isfinite(background_polarisation_dot)
      || std::abs(background_polarisation_dot)
          > kSpiralOrthogonalityTolerance) {
    throw jams::ConfigException(
        settings["cmc_spiral_polarisation"],
        "cmc_spiral_polarisation must be perpendicular to "
        "cmc_spiral_background within ",
        kSpiralOrthogonalityTolerance);
  }

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
    // Plane phase psi_p = phi_0 + 2 pi q_d R_{p,d}, where the integer cell
    // coordinate is R_{p,d} in reduced lattice units.
    const double phase = spiral_phase_
        + kTwoPi * wavevector_component * static_cast<double>(coordinate);
    plane.target_direction = spin_spiral_target(phase);
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

jams::Vec<double, 3> ConstrainedMCSolver::spin_spiral_target(
    const double phase) const {
  // Here n = spiral_background_, e = spiral_polarisation_,
  // alpha = spiral_amplitude_, and phase is psi. Validation guarantees that
  // n and e are orthonormal, so n x e completes the local transverse frame.
  const double background_scale = std::cos(spiral_amplitude_);
  const double transverse_scale = std::sin(spiral_amplitude_);

  jams::Vec<double, 3> transverse_direction;
  if (spiral_profile_ == SpinSpiralProfile::Circular) {
    // u_circular(psi) = e cos(psi) + (n x e) sin(psi), hence
    // t_circular = cos(alpha)n + sin(alpha)u_circular. The three terms form
    // an orthonormal frame, so the exact expression has unit length.
    transverse_direction = std::cos(phase) * spiral_polarisation_
        + std::sin(phase)
            * jams::cross(spiral_background_, spiral_polarisation_);
  } else {
    // A linearly polarised standing wave has transverse displacement
    // u_linear(psi) = e cos(psi). Its length varies with phase, so the final
    // target t_linear = cos(alpha)n + sin(alpha)u_linear must be normalised.
    transverse_direction = std::cos(phase) * spiral_polarisation_;
  }

  // The explicit normalisation implements the linear definition and removes
  // accumulated floating-point error from the circular definition.
  auto target = background_scale * spiral_background_
      + transverse_scale * transverse_direction;
  const double target_norm = jams::norm(target);
  if (!jams::is_finite(target)
      || !std::isfinite(target_norm)
      || target_norm == 0.0) {
    throw std::runtime_error(
        "ConstrainedMCSolver -- spin-spiral target direction is zero or non-finite");
  }
  target /= target_norm;
  return target;
}

void ConstrainedMCSolver::run() {
  // Chooses nspins random spin pairs from the spin system and attempts a
  // Constrained Monte Carlo move on each pair, accepting for either lower
  // energy or with a Boltzmann thermal weighting.
  std::uniform_real_distribution<> uniform_distribution;

  if (is_unrestricted_move_angle_adaptation_active()) {
    validate_move_angle_adaptation_temperature(physics_module_->temperature());
  }

  jams::montecarlo::MonteCarloUniformMove<jams::RandomGeneratorType> uniform_move(&jams::instance().random_generator());
  jams::montecarlo::MonteCarloAngleMove<jams::RandomGeneratorType>   angle_move(&jams::instance().random_generator(), move_angle_sigma_);
  jams::montecarlo::MonteCarloReflectionMove           reflection_move;

  auto uniform_random_number = uniform_distribution(jams::instance().random_generator());
  if (uniform_random_number < move_fraction_uniform_) {
    move_running_acceptance_count_uniform_ += AsselinAlgorithm(uniform_move);
    run_count_uniform_++;
  } else if (uniform_random_number < (move_fraction_uniform_ + move_fraction_angle_)) {
    unsigned attempted_angle_moves = 0;
    const auto accepted_angle_moves = AsselinAlgorithm(
        angle_move, &attempted_angle_moves);
    move_running_acceptance_count_angle_ += accepted_angle_moves;
    run_count_angle_++;
    if (move_angle_adaptation_enabled_ && !move_angle_adaptation_frozen_) {
      move_angle_adaptation_attempted_ += attempted_angle_moves;
      move_angle_adaptation_accepted_ += accepted_angle_moves;
    }
  } else {
    move_running_acceptance_count_reflection_ += AsselinAlgorithm(reflection_move);
    run_count_reflection_++;
  }

  iteration_++;
  time_ = iteration_;

  update_move_angle_adaptation(std::cout);

  if (iteration_ % output_write_steps_ == 0) {
    validate_constraint();

    sum_running_acceptance_statistics();
    output_running_stats_info(std::cout);
    reset_running_statistics();
  }
}

bool ConstrainedMCSolver::is_unrestricted_move_angle_adaptation_active() const {
  return move_angle_adaptation_enabled_
      && !move_angle_adaptation_frozen_
      && !move_angle_burn_in_steps_.has_value();
}

void ConstrainedMCSolver::validate_move_angle_adaptation_temperature(
    const double temperature) const {
  if (is_unrestricted_move_angle_adaptation_active() && temperature != 0.0) {
    throw std::runtime_error(
        "ConstrainedMCSolver -- move_angle_adaptation.burn_in_steps is required "
        "before unrestricted adaptation can run at nonzero temperature");
  }
}

void ConstrainedMCSolver::update_move_angle_adaptation(std::ostream& os) {
  if (!move_angle_adaptation_enabled_ || move_angle_adaptation_frozen_) {
    return;
  }

  const bool interval_boundary =
      iteration_ % move_angle_adaptation_interval_steps_ == 0;
  const bool burn_in_boundary = move_angle_burn_in_steps_.has_value()
      && iteration_ == *move_angle_burn_in_steps_;
  if (!interval_boundary && !burn_in_boundary) {
    return;
  }

  const auto original_flags = os.flags();
  const auto original_precision = os.precision();
  os << std::scientific << std::setprecision(8);

  const double previous_sigma = move_angle_sigma_;
  if (move_angle_adaptation_attempted_ == 0) {
    os << "move_angle_adaptation: step " << iteration_
       << ", attempted 0, accepted 0, acceptance n/a, previous sigma "
       << previous_sigma << ", updated sigma " << move_angle_sigma_
       << ", bound none, update skipped (no angle moves attempted)\n";
  } else {
    const double acceptance = static_cast<double>(move_angle_adaptation_accepted_)
        / static_cast<double>(move_angle_adaptation_attempted_);
    const double proposed_log_sigma = std::log(previous_sigma)
        + move_angle_adaptation_gain_
            * (acceptance - move_angle_target_acceptance_);
    const double min_log_sigma = std::log(move_angle_min_sigma_);
    const double max_log_sigma = std::log(move_angle_max_sigma_);

    const char* bound = "none";
    if (proposed_log_sigma <= min_log_sigma) {
      move_angle_sigma_ = move_angle_min_sigma_;
      bound = "minimum";
    } else if (proposed_log_sigma >= max_log_sigma) {
      move_angle_sigma_ = move_angle_max_sigma_;
      bound = "maximum";
    } else {
      move_angle_sigma_ = std::exp(proposed_log_sigma);
    }

    os << "move_angle_adaptation: step " << iteration_
       << ", attempted " << move_angle_adaptation_attempted_
       << ", accepted " << move_angle_adaptation_accepted_
       << ", acceptance " << acceptance
       << ", previous sigma " << previous_sigma
       << ", updated sigma " << move_angle_sigma_
       << ", bound " << bound << "\n";
  }

  move_angle_adaptation_attempted_ = 0;
  move_angle_adaptation_accepted_ = 0;

  if (burn_in_boundary) {
    move_angle_adaptation_frozen_ = true;
    os << "move_angle_adaptation: burn-in complete at step " << iteration_
       << "; frozen production sigma " << move_angle_sigma_ << "\n";
  }

  os.flags(original_flags);
  os.precision(original_precision);
}

std::vector<jams::output::ColDef> ConstrainedMCSolver::monitor_coordinate_columns() const {
  return {{"step", "steps", jams::output::ColFmt::Integer}};
}

void ConstrainedMCSolver::append_monitor_coordinates(std::vector<double>& values) const {
  values.push_back(iteration());
}

unsigned ConstrainedMCSolver::AsselinAlgorithm(
    const std::function<jams::Vec<double, 3>(jams::Vec<double, 3>)>& trial_spin_move,
    unsigned* moves_attempted) {
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
    if (moves_attempted != nullptr) {
      ++(*moves_attempted);
    }
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

const char* ConstrainedMCSolver::spin_spiral_profile_name() const {
  switch (spiral_profile_) {
    case SpinSpiralProfile::Circular:
      return "circular";
    case SpinSpiralProfile::Linear:
      return "linear";
  }
  return "unknown";
}

void ConstrainedMCSolver::output_initialization_info(std::ostream &os) {
  os << "    constraint mode " << constraint_mode_name() << "\n";
  os << "    constraint type " << constraint_type_name() << "\n";
  os << "    constraint angular tolerance (deg) " << constraint_tolerance_degrees_ << "\n";
  if (constraint_mode_ == ConstraintMode::Global) {
    os << "    constraint angle theta (deg) " << constraint_theta_ << "\n";
    os << "    constraint angle phi (deg) " << constraint_phi_ << "\n";
    os << "    constraint vector " << constraint_vector_ << "\n";
  } else {
    os << "    spiral profile " << spin_spiral_profile_name() << "\n";
    os << "    spiral wavevector " << spiral_wavevector_ << " (cycles per unit cell)\n";
    os << "    zero-wavevector plane constraint "
       << (zero_wavevector_plane_constraint_ ? "yes" : "no") << "\n";
    os << "    spiral background " << spiral_background_ << "\n";
    os << "    spiral polarisation " << spiral_polarisation_ << "\n";
    os << "    spiral amplitude (deg) " << spiral_amplitude_degrees_ << "\n";
    os << "    spiral phase (deg) " << spiral_phase_degrees_ << "\n";
    os << "    spiral reference target "
       << spin_spiral_target(spiral_phase_) << "\n";
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
  os << "    move angle adaptation "
     << (move_angle_adaptation_enabled_ ? "enabled" : "disabled") << "\n";
  if (move_angle_adaptation_enabled_) {
    const auto original_flags = os.flags();
    const auto original_precision = os.precision();
    os << std::scientific << std::setprecision(8);
    os << "      initial sigma " << move_angle_sigma_ << "\n";
    os << "      target acceptance " << move_angle_target_acceptance_ << "\n";
    os << "      interval steps " << move_angle_adaptation_interval_steps_ << "\n";
    os << "      gain " << move_angle_adaptation_gain_ << "\n";
    os << "      sigma bounds " << move_angle_min_sigma_ << " "
       << move_angle_max_sigma_ << "\n";
    if (move_angle_burn_in_steps_.has_value()) {
      os << "      burn-in steps " << *move_angle_burn_in_steps_ << "\n";
    } else {
      os << "      burn-in steps unrestricted (zero-temperature run only)\n";
    }
    os.flags(original_flags);
    os.precision(original_precision);
  }
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

  if (!jams::is_finite(order_parameter)
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
    if (!jams::is_finite(vector) || !std::isfinite(vector_norm) || vector_norm == 0.0) {
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
