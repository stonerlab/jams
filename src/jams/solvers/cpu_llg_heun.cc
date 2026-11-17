// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cpu_llg_heun.h"

#include <cmath>
#include <iostream>
#include <string>

#include "jams/core/globals.h"
#include "jams/interface/config.h"
#include "jams/solvers/llg_rhs.h"

namespace {

void heun_rhs(const double spin[3], jams::Real* field, const jams::Real* noise, const double inv_mu,
              const double gyro_eff, const double alpha, bool store_effective_field,
              double ds_dt[3]) {
  double effective_field[3];
  for (int n = 0; n < 3; ++n) {
    effective_field[n] = noise[n] + field[n] * inv_mu;
    if (store_effective_field) field[n] = static_cast<jams::Real>(effective_field[n]);
  }
  jams::solvers::llg_rhs(spin, effective_field, gyro_eff, alpha, ds_dt);
}

void store_normalised_spin(const double spin[3], double* destination) {
  const double norm_sq = spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2];
  if (norm_sq == 0.0) {
    destination[0] = destination[1] = destination[2] = 0.0;
    return;
  }
  const double inv_norm = 1.0 / std::sqrt(norm_sq);
  for (int n = 0; n < 3; ++n) {
    destination[n] = spin[n] * inv_norm;
  }
}

void heun_stage_1(const double dt, const jams::Real* noise,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff,
                  jams::MultiArray<double, 2>& spins_initial, bool store_effective_field) {
  auto* spins = globals::s.mutable_host_view().data();
  auto* initial_spins = spins_initial.mutable_host_view().data();
  auto* ds_dt = globals::ds_dt.mutable_host_view().data();
  auto* fields = globals::h.mutable_host_view().data();
  const auto* inverse_moments = globals::inv_mus.host_view().data();
  const auto* gyro = gyro_eff.host_data();
  const auto* damping = globals::alpha.host_view().data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    const int offset = 3 * spin_index;
    double k1[3];
    heun_rhs(spins + offset, fields + offset, noise + offset, inverse_moments[spin_index],
             gyro[spin_index], damping[spin_index], store_effective_field, k1);

    double stage_2_spin[3];
    for (int n = 0; n < 3; ++n) {
      const int index = offset + n;
      initial_spins[index] = spins[index];
      ds_dt[index] = 0.5 * k1[n];
      stage_2_spin[n] = spins[index] + dt * k1[n];
    }
    store_normalised_spin(stage_2_spin, spins + offset);
  }
}

void heun_stage_2(const double dt, const jams::Real* noise,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff,
                  const jams::MultiArray<double, 2>& spins_initial, bool store_effective_field) {
  auto* spins = globals::s.mutable_host_view().data();
  const auto* initial_spins = spins_initial.host_data();
  auto* ds_dt = globals::ds_dt.mutable_host_view().data();
  auto* fields = globals::h.mutable_host_view().data();
  const auto* inverse_moments = globals::inv_mus.host_view().data();
  const auto* gyro = gyro_eff.host_data();
  const auto* damping = globals::alpha.host_view().data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    const int offset = 3 * spin_index;
    double k2[3];
    heun_rhs(spins + offset, fields + offset, noise + offset, inverse_moments[spin_index],
             gyro[spin_index], damping[spin_index], store_effective_field, k2);

    double final_spin[3];
    for (int n = 0; n < 3; ++n) {
      const int index = offset + n;
      // Stage 1 stored k1 / 2. Complete the average with k2 / 2.
      ds_dt[index] += 0.5 * k2[n];
      final_spin[n] = initial_spins[index] + dt * ds_dt[index];
    }
    store_normalised_spin(final_spin, spins + offset);
  }
}

}  // namespace

void HeunLLGSolver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);
  initialize_gyro_eff(settings, gyro_eff_);

  const auto thermostat_name =
      jams::config_optional<std::string>(settings, "thermostat", "langevin-white-cpu");
  initialize_stage_thermostat(thermostat_name, jams::thermal_field_coupling(gyro_eff_),
                              1, 1, "held field");

  spins_initial_.resize(globals::num_spins, 3);
}

void HeunLLGSolver::run() {
  const double step_start_time = time_;
  const double dt = step_size_;

  if (!uses_stage_thermostat()) update_thermostat();
  const auto* noise = uses_stage_thermostat() ? sample_stage_thermostat(0) : thermostat_->data();

  compute_fields();
  // With a stage field, keep the Hamiltonian buffer free of noise even
  // when there is no Hamiltonian to refresh it at the next stage.
  heun_stage_1(dt, noise, gyro_eff_, spins_initial_, !uses_stage_thermostat());

  time_ = step_start_time + dt;
  compute_fields();
  if (uses_stage_thermostat()) noise = sample_stage_thermostat(1);
  heun_stage_2(dt, noise, gyro_eff_, spins_initial_, !uses_stage_thermostat());

  advance_time();
}
