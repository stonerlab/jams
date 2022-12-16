// Copyright 2014 Joseph Barker. All rights reserved.
#include <iomanip>

#include <libconfig.h++>
#include "jams/helpers/output.h"

#include "cpu_monte_carlo_constrained.h"

#include "jams/core/jams++.h"
#include "jams/helpers/error.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/helpers/montecarlo.h"
#include "jams/core/lattice.h"
#include "jams/core/physics.h"
#include "jams/helpers/montecarlo.h"

using namespace std;

namespace {
    inline double remap_azimuthal_angle_degrees(double x) {
      // remaps an angle in degrees to the range -180.0 < x <= 180.0
      // this matches the mapping expected for atan2 but avoids the ambiguity
      // of -180.0 == 180.0 which can cause issues for rotation matricies
      return (x = fmod(x + 360.0, 360.0)) > 180.0 ? x - 360.0 : x;
    }
}

void ConstrainedMCSolver::initialize(const libconfig::Setting& settings) {
  do_spin_initial_alignment_ = jams::config_optional(settings, "auto_align", do_spin_initial_alignment_);

  max_steps_ = jams::config_required<int>(settings, "max_steps");
  min_steps_ = jams::config_optional<int>(settings, "min_steps", jams::defaults::solver_min_steps);

  // theta is angle for z to x-y plane from 0 to 180
  constraint_theta_ = jams::config_required<double>(settings, "cmc_constraint_theta");
  // phi is angle in the x-y plane from 0 to 360
  constraint_phi_ = jams::config_required<double>(settings, "cmc_constraint_phi");
  constraint_phi_ = remap_azimuthal_angle_degrees(constraint_phi_);

  move_angle_sigma_        = jams::config_optional<double>(settings, "move_angle_sigma", jams::defaults::solver_monte_carlo_move_sigma);
  output_write_steps_      = jams::config_optional<int>(settings, "output_write_steps",  jams::defaults::monitor_output_steps);

  constraint_vector_       = spherical_to_cartesian_vector(1.0, deg_to_rad(constraint_theta_), deg_to_rad(constraint_phi_));

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

  spin_transformations_.resize(globals::num_spins);
  for (int i = 0; i < globals::num_spins; ++i) {
    spin_transformations_[i] = lattice->material(lattice->atom_material_id(i)).transform;
  }

  output_initialization_info(cout);

  validate_angles();
  validate_rotation_matricies();
  validate_moves();

  if (do_spin_initial_alignment_) {
    align_spins_to_constraint();
  }
  validate_constraint();
}

void ConstrainedMCSolver::run() {
  reset_running_statistics();

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
    output_running_stats_info(cout);
    reset_running_statistics();
  }
}

unsigned ConstrainedMCSolver::AsselinAlgorithm(const std::function<Vec3(Vec3)>& trial_spin_move) {
  using namespace std;
  using namespace globals;

  uniform_real_distribution<> uniform_distribution;

  const double beta = 1.0 / (physics_module_->temperature() * kBoltzmannIU);
  Vec3 magnetisation = total_transformed_magnetization();

  unsigned moves_accepted = 0;

  // we move two spins moving all spins on average is num_spins/2
  for (auto i = 0; i < globals::num_spins/2; ++i) {
    // randomly get two spins s1 != s2
    auto s1 = jams::montecarlo::random_spin_index();

    auto s2 = s1;
    while (s2 == s1) {
      s2 = jams::montecarlo::random_spin_index();
    }

    Vec3 s1_initial         = jams::montecarlo::get_spin(s1);

    Vec3 s1_initial_rotated = rotate_cartesian_to_constraint(s1, s1_initial);

    Vec3 s1_trial           = trial_spin_move(s1_initial);
    Vec3 s1_trial_rotated   = rotate_cartesian_to_constraint(s1, s1_trial);

    Vec3 s2_initial         = jams::montecarlo::get_spin(s2);
    Vec3 s2_initial_rotated = rotate_cartesian_to_constraint(s2, s2_initial);

    // calculate new spin based on contraint mx = my = 0 in the constraint vector reference frame
    Vec3 s2_trial_rotated   = s2_initial_rotated + (s1_initial_rotated - s1_trial_rotated ) * (globals::mus(s1) / globals::mus(s2)) ;

    double ss2 = s2_trial_rotated[0] * s2_trial_rotated[0] + s2_trial_rotated[1] * s2_trial_rotated[1];
    if (ss2 > 1.0) {
      // the rotated spin does not fit on the unit sphere - revert s1 and reject move
      continue;
    }
    // calculate the z-component so that |s2| = 1
    s2_trial_rotated[2] = copysign(sqrt(1.0 - ss2), s2_initial_rotated[2]);

    Vec3 s2_trial = rotate_constraint_to_cartesian(s2, s2_trial_rotated);

    Vec3 delta_m = magnetization_difference(s1, s1_initial, s1_trial, s2, s2_initial, s2_trial);

    Vec3 m_trial_rotated = rotation_matrix_ * (magnetisation + delta_m);

    if (m_trial_rotated[2] < 0.0) {
      // The new magnetization is in the opposite sense - revert s1, reject move
      continue;
    }

    Vec3 m_initial_rotated = rotation_matrix_ * (magnetisation);

    // calculate the Boltzmann weighted probability including the Jacobian factors (see paper)
    double delta_e = energy_difference(s1, s1_initial, s1_trial, s2, s2_initial, s2_trial);
    double jacobian_factor = pow2(m_trial_rotated[2] / m_initial_rotated[2]) * abs(s2_initial_rotated[2] / s2_trial_rotated[2]);
    double probability = min(1.0, exp(-delta_e * beta) * jacobian_factor);

    if (uniform_distribution(jams::instance().random_generator()) > probability) {
      // reject move
      continue;
    }

    // accept move
    jams::montecarlo::set_spin(s1, s1_trial);
    jams::montecarlo::set_spin(s2, s2_trial);

    magnetisation += delta_m;

    moves_accepted++;
  }

  return moves_accepted;
}

double ConstrainedMCSolver::energy_difference(const int &s1, const Vec3 &s1_initial, const Vec3 &s1_trial,
                                              const int &s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const {
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

Vec3 ConstrainedMCSolver::magnetization_difference(const int &s1, const Vec3 &s1_initial, const Vec3 &s1_trial,
                                                   const int &s2, const Vec3 &s2_initial, const Vec3 &s2_trial) const {
  return globals::mus(s1) * spin_transformations_[s1] * (s1_trial - s1_initial)
  + globals::mus(s2) * spin_transformations_[s2] * (s2_trial - s2_initial);
}

Vec3 ConstrainedMCSolver::total_transformed_magnetization() const {
  Vec3 m_total = {0.0, 0.0, 0.0};

  for (auto i = 0; i < globals::num_spins; ++i) {
    m_total += globals::mus(i) * spin_transformations_[i] *
        jams::montecarlo::get_spin(i);
  }

  return m_total;
}

Vec3 ConstrainedMCSolver::rotate_cartesian_to_constraint(const int &i, const Vec3 &spin) const {
  return rotation_matrix_ * (spin_transformations_[i] * spin);
}

Vec3 ConstrainedMCSolver::rotate_constraint_to_cartesian(const int &i, const Vec3 &spin) const {
  return transpose(spin_transformations_[i]) * (inverse_rotation_matrix_ * spin);
}

void ConstrainedMCSolver::output_initialization_info(std::ostream &os) {
  os << "    constraint angle theta (deg) " << constraint_theta_ << "\n";
  os << "    constraint angle phi (deg) " << constraint_phi_ << "\n";
  os << "    constraint vector " << constraint_vector_[0] << " " << constraint_vector_[1] << " " << constraint_vector_[2] << "\n";
  os << "    move_fraction_uniform " << move_fraction_uniform_ << "\n";
  os << "    move_fraction_angle " << move_fraction_angle_ << "\n";
  os << "    move_fraction_reflection " << move_fraction_reflection_ << "\n";
  os << "    move_angle_sigma " << move_angle_sigma_ << "\n";
  os << "    output_write_steps " << output_write_steps_ << "\n";
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

void ConstrainedMCSolver::validate_rotation_matricies() const {
  Vec3 test_unit_vec = {0.0, 0.0, 1.0};
  Vec3 test_forward_vec = rotation_matrix_ * test_unit_vec;
  Vec3 test_back_vec    = inverse_rotation_matrix_ * test_forward_vec;

  cout << "  rotation sanity check\n";
  cout << "    rotate\n";
  cout << "      " << test_unit_vec << " -> " << test_forward_vec << "\n";
  cout << "    back rotate\n";
  cout << "      " << test_forward_vec << " -> " << test_back_vec << "\n";

  for (int n = 0; n < 3; ++n) {
    if (!approximately_equal(test_unit_vec[n], test_back_vec[n], jams::defaults::solver_monte_carlo_constraint_tolerance)) {
      throw std::runtime_error("ConstrainedMCSolver :: rotation sanity check failed");
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
  Vec3 m_total = total_transformed_magnetization();

  const double actual_theta = rad_to_deg(polar_angle(m_total));
  const double actual_phi = rad_to_deg(azimuthal_angle(m_total));

  if (!approximately_equal(actual_theta, constraint_theta_, jams::defaults::solver_monte_carlo_constraint_tolerance)) {
    std::stringstream ss;
    ss << "ConstrainedMCSolver -- theta constraint (" << jams::fmt::decimal << constraint_theta_ << ") violated (" << std::setprecision(10) << std::setw(12) << rad_to_deg(polar_angle(m_total)) << " deg)";
    throw std::runtime_error(ss.str());
  }

  // theta is ~0 or 180 (i.e. it is at a pole) then the phi angle is undefined
  const bool at_pole = approximately_zero(constraint_theta_, DBL_EPSILON) || approximately_equal(constraint_theta_, 180.0, DBL_EPSILON);
  if (!at_pole) {
    if (!approximately_equal(actual_phi, constraint_phi_, jams::defaults::solver_monte_carlo_constraint_tolerance)) {
      std::stringstream ss;
      ss << "ConstrainedMCSolver -- phi constraint (" << jams::fmt::decimal << constraint_phi_ << ") violated ("
         << std::setprecision(10) << std::setw(12) << rad_to_deg(azimuthal_angle(m_total)) << " deg)";
      throw std::runtime_error(ss.str());
    }
  }
}

void ConstrainedMCSolver::validate_angles() const {
  if (constraint_theta_ < 0 || constraint_theta_ > 180.0) {
    throw std::runtime_error(
        "ConstrainedMCSolver -- theta ( " + to_string(constraint_theta_) + " ) is out of range (0 <= theta <= 180)");
  }

  if ( constraint_phi_ <= -180.0 || constraint_phi_ > 180.0) {
    throw std::runtime_error(
        "ConstrainedMCSolver -- phi ( " + to_string(constraint_phi_) + " ) is out of range (-180 <= phi <= 180)");
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
  auto M = total_transformed_magnetization();

  auto rotation = rotation_matrix_between_vectors(M, constraint_vector_);

  for (auto i = 0; i < globals::num_spins; ++i) {
    Vec3 snew = rotation * jams::montecarlo::get_spin(i);
    for (auto j : {0, 1, 2}) {
      globals::s(i, j) = snew[j];
    }
  }
}
