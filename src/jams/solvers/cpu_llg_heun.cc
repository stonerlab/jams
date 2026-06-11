// Copyright 2014 Joseph Barker. All rights reserved.

#include "cpu_llg_heun.h"

#include <cmath>
#include <jams/interface/config.h>
#include <jams/helpers/maths.h>

#include "jams/core/globals.h"
#include "jams/core/physics.h"

void HeunLLGSolver::initialize(const libconfig::Setting& settings) {
  // convert input in seconds to picoseconds for internal units
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  std::cout << "\ntimestep (ps) " << step_size_ << "\n";
  std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  s_old_.resize(globals::num_spins, 3);

  initialize_gyro_eff(settings, gyro_eff_);

  const std::string thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", "langevin-white-cpu");
  register_thermostat(Thermostat::create(thermostat_name, this->time_step()));
  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";
}

void HeunLLGSolver::run() {
  double t0 = time_;

  // copy the spin configuration at the start of the step
  s_old_ = globals::s;

  update_thermostat();

  Solver::compute_fields();

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      globals::h(i, j) = thermostat_->field(i, j) + globals::h(i, j) / globals::mus(i);
    }
  }

  for (auto i = 0; i < globals::num_spins; ++i) {
    jams::Vec<double, 3> spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
    jams::Vec<double, 3> field = {globals::h(i,0), globals::h(i,1), globals::h(i,2)};

    jams::Vec<double, 3> rhs = -gyro_eff_(i) * (jams::cross(spin, field) + globals::alpha(i) * jams::cross(spin, (jams::cross(spin, field))));

    for (auto j = 0; j < 3; ++j) {
      globals::ds_dt(i, j) = 0.5 * rhs[j];
    }

    spin = jams::unit_vector(spin + step_size_ * rhs);

     for (auto j = 0; j < 3; ++j) {
       globals::s(i, j) = spin[j];
    }

  }

  double mid_time_step = step_size_;
  time_ = t0 + mid_time_step;

  Solver::compute_fields();

  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      globals::h(i, j) = thermostat_->field(i, j) + globals::h(i, j) / globals::mus(i);
    }
  }

  for (auto i = 0; i < globals::num_spins; ++i) {
    jams::Vec<double, 3> spin = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
    jams::Vec<double, 3> spin_old = {s_old_(i,0), s_old_(i,1), s_old_(i,2)};

    jams::Vec<double, 3> field = {globals::h(i,0), globals::h(i,1), globals::h(i,2)};
    jams::Vec<double, 3> rhs = -gyro_eff_(i) * (jams::cross(spin, field) + globals::alpha(i) * jams::cross(spin, (jams::cross(spin, field))));

    for (auto j = 0; j < 3; ++j) {
      globals::ds_dt(i, j) = globals::ds_dt(i, j) + 0.5 * rhs[j];
    }

    jams::Vec<double, 3> ds = {globals::ds_dt(i, 0), globals::ds_dt(i, 1) , globals::ds_dt(i, 2)};

    spin = jams::unit_vector(spin_old + step_size_ * ds);

    for (auto j = 0; j < 3; ++j) {
      globals::s(i, j) = spin[j];
    }

  }

  iteration_++;
  time_ = iteration_ * step_size_;
}
