// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cpu_llg_rkmk.h"

#include <iostream>
#include <string>

#include "jams/interface/config.h"

#include "jams/core/globals.h"
#include "jams/solvers/llg_rkmk_thermal_kick.h"

namespace {

using jams::solvers::rkmk::ThermalKick;
using jams::solvers::rkmk::make_thermal_kick;

void apply_noise_step(const double dt, const jams::MultiArray<jams::Real, 1>& gyro_eff,
                      const jams::Real* noise_values, const ThermalKick& kick,
                      const bool in_parallel = false) {
  double* spin_values = nullptr;
  const jams::Real* alpha_values = nullptr;
  const jams::Real* gyro_eff_values = nullptr;

#if HAS_OMP
  if (in_parallel) {
#pragma omp single copyprivate(spin_values, alpha_values, gyro_eff_values)
    {
      spin_values = globals::s.data();
      alpha_values = globals::alpha.host_data();
      gyro_eff_values = gyro_eff.host_data();
    }
  } else
#endif
  {
    auto spin_view = globals::s.mutable_host_view();
    const auto alpha_view = globals::alpha.host_view();
    const auto gyro_eff_view = gyro_eff.host_view();
    spin_values = spin_view.data();
    alpha_values = alpha_view.data();
    gyro_eff_values = gyro_eff_view.data();
  }

  const auto update_spin = [&](const int spin_index) {
    const int offset = 3 * spin_index;
    const double s[3] = {spin_values[offset], spin_values[offset + 1], spin_values[offset + 2]};
    double s_out[3];
    kick.apply(s, noise_values + offset, gyro_eff_values[spin_index],
               alpha_values[spin_index], dt, spin_index, s_out);

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  };

#if HAS_OMP
  if (in_parallel) {
#pragma omp for schedule(static)
    for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      update_spin(spin_index);
    }
    return;
  }
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    update_spin(spin_index);
  }
}

void rkmk_effective_field(const jams::Real* fields, const jams::Real* stage_noise,
                          int offset, double inv_mu, double h[3]) {
  for (int n = 0; n < 3; ++n) {
    h[n] = static_cast<double>(fields[offset + n]) * inv_mu;
    if (stage_noise) h[n] += stage_noise[offset + n];
  }
}

void rkmk2_stage_1(const double dt, jams::MultiArray<double, 2>& spins_initial,
                   jams::MultiArray<double, 2>& phi,
                   const jams::MultiArray<jams::Real, 1>& gyro_eff,
                   const jams::Real* stage_noise,
                   const bool in_parallel = false) {
  const jams::Real* field_values = nullptr;
  const jams::Real* inv_mu_values = nullptr;
  const jams::Real* alpha_values = nullptr;
  const jams::Real* gyro_eff_values = nullptr;
  double* spin_values = nullptr;
  double* initial_spins = nullptr;
  double* phi_values = nullptr;

#if HAS_OMP
  if (in_parallel) {
#pragma omp single copyprivate(field_values, inv_mu_values, alpha_values, gyro_eff_values, \
                                   spin_values, initial_spins, phi_values)
    {
      field_values = globals::h.host_data();
      inv_mu_values = globals::inv_mus.host_data();
      alpha_values = globals::alpha.host_data();
      gyro_eff_values = gyro_eff.host_data();
      spin_values = globals::s.data();
      initial_spins = spins_initial.data();
      phi_values = phi.data();
    }
  } else
#endif
  {
    const auto field_view = globals::h.host_view();
    const auto inv_mu_view = globals::inv_mus.host_view();
    const auto alpha_view = globals::alpha.host_view();
    const auto gyro_eff_view = gyro_eff.host_view();
    auto spin_view = globals::s.mutable_host_view();
    auto initial_spin_view = spins_initial.mutable_host_view();
    auto phi_view = phi.mutable_host_view();

    field_values = field_view.data();
    inv_mu_values = inv_mu_view.data();
    alpha_values = alpha_view.data();
    gyro_eff_values = gyro_eff_view.data();
    spin_values = spin_view.data();
    initial_spins = initial_spin_view.data();
    phi_values = phi_view.data();
  }

  const auto update_spin = [&](const int spin_index) {
    const int offset = 3 * spin_index;
    const double inv_mu = static_cast<double>(inv_mu_values[spin_index]);
    double h[3];
    rkmk_effective_field(field_values, stage_noise, offset, inv_mu, h);
    const double s[3] = {spin_values[offset], spin_values[offset + 1], spin_values[offset + 2]};

    initial_spins[offset] = s[0];
    initial_spins[offset + 1] = s[1];
    initial_spins[offset + 2] = s[2];

    double omega[3];
    jams::solvers::rkmk::omega_llg(s, h, gyro_eff_values[spin_index], alpha_values[spin_index],
                                   omega);

    const double phi_i[3] = {0.5 * dt * omega[0], 0.5 * dt * omega[1], 0.5 * dt * omega[2]};
    phi_values[offset] = phi_i[0];
    phi_values[offset + 1] = phi_i[1];
    phi_values[offset + 2] = phi_i[2];

    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(phi_i, s, s_out);

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  };

#if HAS_OMP
  if (in_parallel) {
#pragma omp for schedule(static)
    for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      update_spin(spin_index);
    }
    return;
  }
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    update_spin(spin_index);
  }
}

void rkmk2_stage_2(const double dt, const double noise_dt,
                   const jams::MultiArray<double, 2>& spins_initial,
                   const jams::MultiArray<double, 2>& phi,
                   const jams::MultiArray<jams::Real, 1>& gyro_eff,
                   const jams::Real* stage_noise, const jams::Real* split_noise,
                   const ThermalKick& kick,
                   const bool in_parallel = false) {
  const jams::Real* field_values = nullptr;
  const jams::Real* inv_mu_values = nullptr;
  const jams::Real* alpha_values = nullptr;
  const jams::Real* gyro_eff_values = nullptr;
  const double* initial_spins = nullptr;
  const double* phi_values = nullptr;
  double* spin_values = nullptr;

#if HAS_OMP
  if (in_parallel) {
#pragma omp single copyprivate(field_values, inv_mu_values, alpha_values, gyro_eff_values, \
                                   initial_spins, phi_values, spin_values)
    {
      field_values = globals::h.host_data();
      inv_mu_values = globals::inv_mus.host_data();
      alpha_values = globals::alpha.host_data();
      gyro_eff_values = gyro_eff.host_data();
      initial_spins = spins_initial.host_data();
      phi_values = phi.host_data();
      spin_values = globals::s.data();
    }
  } else
#endif
  {
    const auto field_view = globals::h.host_view();
    const auto inv_mu_view = globals::inv_mus.host_view();
    const auto alpha_view = globals::alpha.host_view();
    const auto gyro_eff_view = gyro_eff.host_view();
    const auto initial_spin_view = spins_initial.host_view();
    const auto phi_view = phi.host_view();
    auto spin_view = globals::s.mutable_host_view();

    field_values = field_view.data();
    inv_mu_values = inv_mu_view.data();
    alpha_values = alpha_view.data();
    gyro_eff_values = gyro_eff_view.data();
    initial_spins = initial_spin_view.data();
    phi_values = phi_view.data();
    spin_values = spin_view.data();
  }

  const auto update_spin = [&](const int spin_index) {
    const int offset = 3 * spin_index;
    const double inv_mu = static_cast<double>(inv_mu_values[spin_index]);
    double h[3];
    rkmk_effective_field(field_values, stage_noise, offset, inv_mu, h);
    const double s_step[3] = {spin_values[offset], spin_values[offset + 1],
                              spin_values[offset + 2]};

    double omega[3];
    jams::solvers::rkmk::omega_llg(s_step, h, gyro_eff_values[spin_index], alpha_values[spin_index],
                                   omega);

    const double v2[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const double phi_i[3] = {phi_values[offset], phi_values[offset + 1], phi_values[offset + 2]};

    double k[3];
    jams::solvers::rkmk::dexp_inv_so3(phi_i, v2, k);

    const double s0[3] = {initial_spins[offset], initial_spins[offset + 1],
                          initial_spins[offset + 2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(k, s0, s_out);

    if (split_noise) {
      kick.apply(s_out, split_noise + offset, gyro_eff_values[spin_index],
                 alpha_values[spin_index], noise_dt, spin_index, s_out);
    }

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  };

#if HAS_OMP
  if (in_parallel) {
#pragma omp for schedule(static)
    for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
      update_spin(spin_index);
    }
    return;
  }
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    update_spin(spin_index);
  }
}

void rkmk4_stage_1(const double dt, jams::MultiArray<double, 2>& spins_initial,
                   jams::MultiArray<double, 2>& k1,
                   const jams::MultiArray<jams::Real, 1>& gyro_eff,
                   const jams::Real* stage_noise) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  auto spin_view = globals::s.mutable_host_view();
  auto initial_spin_view = spins_initial.mutable_host_view();
  auto k1_view = k1.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  auto* spin_values = spin_view.data();
  auto* initial_spins = initial_spin_view.data();
  auto* k1_values = k1_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    const int offset = 3 * spin_index;
    const double inv_mu = static_cast<double>(inv_mu_values[spin_index]);
    double h[3];
    rkmk_effective_field(field_values, stage_noise, offset, inv_mu, h);
    const double s[3] = {spin_values[offset], spin_values[offset + 1], spin_values[offset + 2]};

    initial_spins[offset] = s[0];
    initial_spins[offset + 1] = s[1];
    initial_spins[offset + 2] = s[2];

    double omega[3];
    jams::solvers::rkmk::omega_llg(s, h, gyro_eff_values[spin_index], alpha_values[spin_index],
                                   omega);

    const double k1_i[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    k1_values[offset] = k1_i[0];
    k1_values[offset + 1] = k1_i[1];
    k1_values[offset + 2] = k1_i[2];

    const double phi[3] = {0.5 * k1_i[0], 0.5 * k1_i[1], 0.5 * k1_i[2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(phi, s, s_out);

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  }
}

void rkmk4_stage_2(const double dt, const jams::MultiArray<double, 2>& spins_initial,
                   const jams::MultiArray<double, 2>& k1, jams::MultiArray<double, 2>& k2,
                   const jams::MultiArray<jams::Real, 1>& gyro_eff,
                   const jams::Real* stage_noise) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  const auto initial_spin_view = spins_initial.host_view();
  const auto k1_view = k1.host_view();
  auto k2_view = k2.mutable_host_view();
  auto spin_view = globals::s.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  const auto* initial_spins = initial_spin_view.data();
  const auto* k1_values = k1_view.data();
  auto* k2_values = k2_view.data();
  auto* spin_values = spin_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    const int offset = 3 * spin_index;
    const double inv_mu = static_cast<double>(inv_mu_values[spin_index]);
    double h[3];
    rkmk_effective_field(field_values, stage_noise, offset, inv_mu, h);
    const double s_step[3] = {spin_values[offset], spin_values[offset + 1],
                              spin_values[offset + 2]};

    double omega[3];
    jams::solvers::rkmk::omega_llg(s_step, h, gyro_eff_values[spin_index], alpha_values[spin_index],
                                   omega);
    const double v2[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const double phi[3] = {0.5 * k1_values[offset], 0.5 * k1_values[offset + 1],
                           0.5 * k1_values[offset + 2]};

    double k2_i[3];
    jams::solvers::rkmk::dexp_inv_so3(phi, v2, k2_i);
    k2_values[offset] = k2_i[0];
    k2_values[offset + 1] = k2_i[1];
    k2_values[offset + 2] = k2_i[2];

    const double s0[3] = {initial_spins[offset], initial_spins[offset + 1],
                          initial_spins[offset + 2]};
    const double phi2[3] = {0.5 * k2_i[0], 0.5 * k2_i[1], 0.5 * k2_i[2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(phi2, s0, s_out);

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  }
}

void rkmk4_stage_3(const double dt, const jams::MultiArray<double, 2>& spins_initial,
                   const jams::MultiArray<double, 2>& k2, jams::MultiArray<double, 2>& k3,
                   const jams::MultiArray<jams::Real, 1>& gyro_eff,
                   const jams::Real* stage_noise) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  const auto initial_spin_view = spins_initial.host_view();
  const auto k2_view = k2.host_view();
  auto k3_view = k3.mutable_host_view();
  auto spin_view = globals::s.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  const auto* initial_spins = initial_spin_view.data();
  const auto* k2_values = k2_view.data();
  auto* k3_values = k3_view.data();
  auto* spin_values = spin_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    const int offset = 3 * spin_index;
    const double inv_mu = static_cast<double>(inv_mu_values[spin_index]);
    double h[3];
    rkmk_effective_field(field_values, stage_noise, offset, inv_mu, h);
    const double s_step[3] = {spin_values[offset], spin_values[offset + 1],
                              spin_values[offset + 2]};

    double omega[3];
    jams::solvers::rkmk::omega_llg(s_step, h, gyro_eff_values[spin_index], alpha_values[spin_index],
                                   omega);
    const double v3[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const double phi[3] = {0.5 * k2_values[offset], 0.5 * k2_values[offset + 1],
                           0.5 * k2_values[offset + 2]};

    double k3_i[3];
    jams::solvers::rkmk::dexp_inv_so3(phi, v3, k3_i);
    k3_values[offset] = k3_i[0];
    k3_values[offset + 1] = k3_i[1];
    k3_values[offset + 2] = k3_i[2];

    const double s0[3] = {initial_spins[offset], initial_spins[offset + 1],
                          initial_spins[offset + 2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(k3_i, s0, s_out);

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  }
}

void rkmk4_stage_4(const double dt, const double noise_dt,
                   const jams::MultiArray<double, 2>& spins_initial,
                   const jams::MultiArray<double, 2>& k1, const jams::MultiArray<double, 2>& k2,
                   const jams::MultiArray<double, 2>& k3,
                   const jams::MultiArray<jams::Real, 1>& gyro_eff,
                   const jams::Real* stage_noise, const jams::Real* split_noise,
                   const ThermalKick& kick) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  const auto initial_spin_view = spins_initial.host_view();
  const auto k1_view = k1.host_view();
  const auto k2_view = k2.host_view();
  const auto k3_view = k3.host_view();
  auto spin_view = globals::s.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  const auto* initial_spins = initial_spin_view.data();
  const auto* k1_values = k1_view.data();
  const auto* k2_values = k2_view.data();
  const auto* k3_values = k3_view.data();
  auto* spin_values = spin_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int spin_index = 0; spin_index < globals::num_spins; ++spin_index) {
    const int offset = 3 * spin_index;
    const double inv_mu = static_cast<double>(inv_mu_values[spin_index]);
    double h[3];
    rkmk_effective_field(field_values, stage_noise, offset, inv_mu, h);
    const double s_step[3] = {spin_values[offset], spin_values[offset + 1],
                              spin_values[offset + 2]};

    double omega[3];
    jams::solvers::rkmk::omega_llg(s_step, h, gyro_eff_values[spin_index], alpha_values[spin_index],
                                   omega);
    const double v4[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const double phi[3] = {k3_values[offset], k3_values[offset + 1], k3_values[offset + 2]};

    double k4[3];
    jams::solvers::rkmk::dexp_inv_so3(phi, v4, k4);

    const double k[3] = {
        (k1_values[offset] + 2.0 * k2_values[offset] + 2.0 * k3_values[offset] + k4[0]) / 6.0,
        (k1_values[offset + 1] + 2.0 * k2_values[offset + 1] + 2.0 * k3_values[offset + 1] +
         k4[1]) /
            6.0,
        (k1_values[offset + 2] + 2.0 * k2_values[offset + 2] + 2.0 * k3_values[offset + 2] +
         k4[2]) /
            6.0};

    const double s0[3] = {initial_spins[offset], initial_spins[offset + 1],
                          initial_spins[offset + 2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(k, s0, s_out);

    if (split_noise) {
      kick.apply(s_out, split_noise + offset, gyro_eff_values[spin_index],
                 alpha_values[spin_index], noise_dt, spin_index, s_out);
    }

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  }
}

}  // namespace

void RKMK2LLGSolver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  initialize_gyro_eff(settings, gyro_eff_);

  const std::string thermostat_name =
      jams::config_optional<std::string>(settings, "thermostat", "langevin-white-cpu");
  initialize_stage_thermostat(thermostat_name, jams::thermal_field_coupling(gyro_eff_),
                              2, 2, "split kicks");

  spins_initial_.resize(globals::num_spins, 3);
  phi_.resize(globals::num_spins, 3);
}

void RKMK2LLGSolver::run() {
  if (supports_persistent_parallel_step()) {
    run_persistent_parallel_step();
    return;
  }

  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;

  ThermalKick kick;
  if (!uses_stage_thermostat()) {
    update_thermostat();
    kick = make_thermal_kick(*thermostat_, globals::inv_mus, half_dt,
                            uses_automatic_thermostat_sampling());
    apply_noise_step(half_dt, gyro_eff_, thermostat_->data(), kick);
  }

  compute_fields();
  rkmk2_stage_1(dt, spins_initial_, phi_, gyro_eff_, sample_stage_thermostat(0));

  time_ = step_start_time + half_dt;

  compute_fields();

  const jams::Real* split_noise = nullptr;
  if (!uses_stage_thermostat()) {
    update_thermostat();
    split_noise = thermostat_->data();
  }
  rkmk2_stage_2(dt, half_dt, spins_initial_, phi_, gyro_eff_,
                sample_stage_thermostat(1), split_noise, kick);
  sample_stage_thermostat(2);

  advance_time();
}

void RKMK4LLGSolver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  initialize_gyro_eff(settings, gyro_eff_);

  const std::string thermostat_name =
      jams::config_optional<std::string>(settings, "thermostat", "langevin-white-cpu");
  initialize_stage_thermostat(thermostat_name, jams::thermal_field_coupling(gyro_eff_),
                              2, 2, "split kicks");

  spins_initial_.resize(globals::num_spins, 3);
  k1_.resize(globals::num_spins, 3);
  k2_.resize(globals::num_spins, 3);
  k3_.resize(globals::num_spins, 3);
}

void RKMK4LLGSolver::run() {
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;

  ThermalKick kick;
  if (!uses_stage_thermostat()) {
    update_thermostat();
    kick = make_thermal_kick(*thermostat_, globals::inv_mus, half_dt,
                            uses_automatic_thermostat_sampling());
    apply_noise_step(half_dt, gyro_eff_, thermostat_->data(), kick);
  }

  compute_fields();
  rkmk4_stage_1(dt, spins_initial_, k1_, gyro_eff_, sample_stage_thermostat(0));

  time_ = step_start_time + half_dt;

  compute_fields();
  const auto* midpoint_noise = sample_stage_thermostat(1);
  rkmk4_stage_2(dt, spins_initial_, k1_, k2_, gyro_eff_, midpoint_noise);

  compute_fields();
  rkmk4_stage_3(dt, spins_initial_, k2_, k3_, gyro_eff_, midpoint_noise);

  time_ = step_start_time + dt;

  compute_fields();

  const jams::Real* split_noise = nullptr;
  if (!uses_stage_thermostat()) {
    update_thermostat();
    split_noise = thermostat_->data();
  }
  rkmk4_stage_4(dt, half_dt, spins_initial_, k1_, k2_, k3_, gyro_eff_,
                sample_stage_thermostat(2), split_noise, kick);

  advance_time();
}

bool RKMK2LLGSolver::supports_persistent_parallel_step() const {
#if HAS_OMP
  return !uses_stage_thermostat() && thermostat_ && thermostat_->supports_update_in_parallel() &&
         !(uses_automatic_thermostat_sampling() && thermostat_->supplies_white_field()) &&
         supports_compute_fields_in_parallel();
#else
  return false;
#endif
}

void RKMK2LLGSolver::run_persistent_parallel_step() {
#if HAS_OMP
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;

#pragma omp parallel
  {
    const jams::Real* noise_values = nullptr;

    thermostat_->update_in_parallel();
#pragma omp single copyprivate(noise_values)
    {
      noise_values = thermostat_->data();
    }
    apply_noise_step(half_dt, gyro_eff_, noise_values, {}, true);

    compute_fields_in_parallel();
    rkmk2_stage_1(dt, spins_initial_, phi_, gyro_eff_, nullptr, true);

#pragma omp single
    {
      time_ = step_start_time + half_dt;
    }

    compute_fields_in_parallel();

    thermostat_->update_in_parallel();
#pragma omp single copyprivate(noise_values)
    {
      noise_values = thermostat_->data();
    }
    rkmk2_stage_2(dt, half_dt, spins_initial_, phi_, gyro_eff_, nullptr, noise_values, {}, true);

#pragma omp single
    {
      advance_time();
    }
  }
#else
  run();
#endif
}
