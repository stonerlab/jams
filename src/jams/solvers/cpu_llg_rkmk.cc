// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cpu_llg_rkmk.h"

#include <iostream>
#include <string>

#include <jams/interface/config.h>

#include "jams/core/globals.h"
#include "jams/solvers/llg_rkmk_functions.h"

namespace {

void apply_noise_step_rodrigues(
    const double dt,
    const jams::MultiArray<jams::Real, 1>& gyro_eff,
    const jams::Real* const noise_values,
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

  const auto update_spin = [&](const int i) {
      const int offset = 3 * i;
      const double s[3] = {
          spin_values[offset],
          spin_values[offset + 1],
          spin_values[offset + 2]};
      const jams::Real h_noise[3] = {
          noise_values[offset],
          noise_values[offset + 1],
          noise_values[offset + 2]};

      double s_out[3];
      jams::solvers::rkmk::noise_step_rodrigues(
          s,
          h_noise,
          gyro_eff_values[i],
          alpha_values[i],
          dt,
          s_out);

      spin_values[offset] = s_out[0];
      spin_values[offset + 1] = s_out[1];
      spin_values[offset + 2] = s_out[2];
  };

#if HAS_OMP
  if (in_parallel) {
#pragma omp for schedule(static)
    for (int i = 0; i < globals::num_spins; ++i) {
      update_spin(i);
    }
    return;
  }
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < globals::num_spins; ++i) {
    update_spin(i);
  }
}

void rkmk2_step_1(const double dt,
                  jams::MultiArray<double, 2>& s_init,
                  jams::MultiArray<double, 2>& phi,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff,
                  const bool in_parallel = false) {
  const jams::Real* field_values = nullptr;
  const jams::Real* inv_mu_values = nullptr;
  const jams::Real* alpha_values = nullptr;
  const jams::Real* gyro_eff_values = nullptr;
  double* spin_values = nullptr;
  double* s_init_values = nullptr;
  double* phi_values = nullptr;

#if HAS_OMP
  if (in_parallel) {
#pragma omp single copyprivate(field_values, inv_mu_values, alpha_values, gyro_eff_values, spin_values, s_init_values, phi_values)
    {
      field_values = globals::h.host_data();
      inv_mu_values = globals::inv_mus.host_data();
      alpha_values = globals::alpha.host_data();
      gyro_eff_values = gyro_eff.host_data();
      spin_values = globals::s.data();
      s_init_values = s_init.data();
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
    auto s_init_view = s_init.mutable_host_view();
    auto phi_view = phi.mutable_host_view();

    field_values = field_view.data();
    inv_mu_values = inv_mu_view.data();
    alpha_values = alpha_view.data();
    gyro_eff_values = gyro_eff_view.data();
    spin_values = spin_view.data();
    s_init_values = s_init_view.data();
    phi_values = phi_view.data();
  }

  const auto update_spin = [&](const int i) {
      const int offset = 3 * i;
      const double inv_mu = static_cast<double>(inv_mu_values[i]);
      const double h[3] = {
          static_cast<double>(field_values[offset]) * inv_mu,
          static_cast<double>(field_values[offset + 1]) * inv_mu,
          static_cast<double>(field_values[offset + 2]) * inv_mu};
      const double s[3] = {
          spin_values[offset],
          spin_values[offset + 1],
          spin_values[offset + 2]};

      s_init_values[offset] = s[0];
      s_init_values[offset + 1] = s[1];
      s_init_values[offset + 2] = s[2];

      double omega[3];
      jams::solvers::rkmk::omega_llg(
          s, h, gyro_eff_values[i], alpha_values[i], omega);

      const double phi_i[3] = {
          0.5 * dt * omega[0],
          0.5 * dt * omega[1],
          0.5 * dt * omega[2]};
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
    for (int i = 0; i < globals::num_spins; ++i) {
      update_spin(i);
    }
    return;
  }
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < globals::num_spins; ++i) {
    update_spin(i);
  }
}

void rkmk2_step_2(const double dt,
                  const double noise_dt,
                  const jams::MultiArray<double, 2>& s_init,
                  const jams::MultiArray<double, 2>& phi,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff,
                  const jams::Real* const noise_values,
                  const bool in_parallel = false) {
  const jams::Real* field_values = nullptr;
  const jams::Real* inv_mu_values = nullptr;
  const jams::Real* alpha_values = nullptr;
  const jams::Real* gyro_eff_values = nullptr;
  const double* s_init_values = nullptr;
  const double* phi_values = nullptr;
  double* spin_values = nullptr;

#if HAS_OMP
  if (in_parallel) {
#pragma omp single copyprivate(field_values, inv_mu_values, alpha_values, gyro_eff_values, s_init_values, phi_values, spin_values)
    {
      field_values = globals::h.host_data();
      inv_mu_values = globals::inv_mus.host_data();
      alpha_values = globals::alpha.host_data();
      gyro_eff_values = gyro_eff.host_data();
      s_init_values = s_init.host_data();
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
    const auto s_init_view = s_init.host_view();
    const auto phi_view = phi.host_view();
    auto spin_view = globals::s.mutable_host_view();

    field_values = field_view.data();
    inv_mu_values = inv_mu_view.data();
    alpha_values = alpha_view.data();
    gyro_eff_values = gyro_eff_view.data();
    s_init_values = s_init_view.data();
    phi_values = phi_view.data();
    spin_values = spin_view.data();
  }

  const auto update_spin = [&](const int i) {
      const int offset = 3 * i;
      const double inv_mu = static_cast<double>(inv_mu_values[i]);
      const double h[3] = {
          static_cast<double>(field_values[offset]) * inv_mu,
          static_cast<double>(field_values[offset + 1]) * inv_mu,
          static_cast<double>(field_values[offset + 2]) * inv_mu};
      const double s_step[3] = {
          spin_values[offset],
          spin_values[offset + 1],
          spin_values[offset + 2]};

      double omega[3];
      jams::solvers::rkmk::omega_llg(
          s_step, h, gyro_eff_values[i], alpha_values[i], omega);

      const double v2[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
      const double phi_i[3] = {
          phi_values[offset],
          phi_values[offset + 1],
          phi_values[offset + 2]};

      double k[3];
      jams::solvers::rkmk::dexp_inv_so3(phi_i, v2, k);

      const double s0[3] = {
          s_init_values[offset],
          s_init_values[offset + 1],
          s_init_values[offset + 2]};
      double s_out[3];
      jams::solvers::rkmk::rodrigues_rotate(k, s0, s_out);

      const jams::Real h_noise[3] = {
          noise_values[offset],
          noise_values[offset + 1],
          noise_values[offset + 2]};
      double s_noisy[3];
      jams::solvers::rkmk::noise_step_rodrigues(
          s_out,
          h_noise,
          gyro_eff_values[i],
          alpha_values[i],
          noise_dt,
          s_noisy);

      spin_values[offset] = s_noisy[0];
      spin_values[offset + 1] = s_noisy[1];
      spin_values[offset + 2] = s_noisy[2];
  };

#if HAS_OMP
  if (in_parallel) {
#pragma omp for schedule(static)
    for (int i = 0; i < globals::num_spins; ++i) {
      update_spin(i);
    }
    return;
  }
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < globals::num_spins; ++i) {
    update_spin(i);
  }
}

void rkmk4_step_1(const double dt,
                  jams::MultiArray<double, 2>& s_init,
                  jams::MultiArray<double, 2>& k1,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  auto spin_view = globals::s.mutable_host_view();
  auto s_init_view = s_init.mutable_host_view();
  auto k1_view = k1.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  auto* spin_values = spin_view.data();
  auto* s_init_values = s_init_view.data();
  auto* k1_values = k1_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < globals::num_spins; ++i) {
    const int offset = 3 * i;
    const double inv_mu = static_cast<double>(inv_mu_values[i]);
    const double h[3] = {
        static_cast<double>(field_values[offset]) * inv_mu,
        static_cast<double>(field_values[offset + 1]) * inv_mu,
        static_cast<double>(field_values[offset + 2]) * inv_mu};
    const double s[3] = {
        spin_values[offset],
        spin_values[offset + 1],
        spin_values[offset + 2]};

    s_init_values[offset] = s[0];
    s_init_values[offset + 1] = s[1];
    s_init_values[offset + 2] = s[2];

    double omega[3];
    jams::solvers::rkmk::omega_llg(
        s, h, gyro_eff_values[i], alpha_values[i], omega);

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

void rkmk4_step_2(const double dt,
                  const jams::MultiArray<double, 2>& s_init,
                  const jams::MultiArray<double, 2>& k1,
                  jams::MultiArray<double, 2>& k2,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  const auto s_init_view = s_init.host_view();
  const auto k1_view = k1.host_view();
  auto k2_view = k2.mutable_host_view();
  auto spin_view = globals::s.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  const auto* s_init_values = s_init_view.data();
  const auto* k1_values = k1_view.data();
  auto* k2_values = k2_view.data();
  auto* spin_values = spin_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < globals::num_spins; ++i) {
    const int offset = 3 * i;
    const double inv_mu = static_cast<double>(inv_mu_values[i]);
    const double h[3] = {
        static_cast<double>(field_values[offset]) * inv_mu,
        static_cast<double>(field_values[offset + 1]) * inv_mu,
        static_cast<double>(field_values[offset + 2]) * inv_mu};
    const double s_step[3] = {
        spin_values[offset],
        spin_values[offset + 1],
        spin_values[offset + 2]};

    double omega[3];
    jams::solvers::rkmk::omega_llg(
        s_step, h, gyro_eff_values[i], alpha_values[i], omega);
    const double v2[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const double phi[3] = {
        0.5 * k1_values[offset],
        0.5 * k1_values[offset + 1],
        0.5 * k1_values[offset + 2]};

    double k2_i[3];
    jams::solvers::rkmk::dexp_inv_so3(phi, v2, k2_i);
    k2_values[offset] = k2_i[0];
    k2_values[offset + 1] = k2_i[1];
    k2_values[offset + 2] = k2_i[2];

    const double s0[3] = {
        s_init_values[offset],
        s_init_values[offset + 1],
        s_init_values[offset + 2]};
    const double phi2[3] = {0.5 * k2_i[0], 0.5 * k2_i[1], 0.5 * k2_i[2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(phi2, s0, s_out);

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  }
}

void rkmk4_step_3(const double dt,
                  const jams::MultiArray<double, 2>& s_init,
                  const jams::MultiArray<double, 2>& k2,
                  jams::MultiArray<double, 2>& k3,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  const auto s_init_view = s_init.host_view();
  const auto k2_view = k2.host_view();
  auto k3_view = k3.mutable_host_view();
  auto spin_view = globals::s.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  const auto* s_init_values = s_init_view.data();
  const auto* k2_values = k2_view.data();
  auto* k3_values = k3_view.data();
  auto* spin_values = spin_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < globals::num_spins; ++i) {
    const int offset = 3 * i;
    const double inv_mu = static_cast<double>(inv_mu_values[i]);
    const double h[3] = {
        static_cast<double>(field_values[offset]) * inv_mu,
        static_cast<double>(field_values[offset + 1]) * inv_mu,
        static_cast<double>(field_values[offset + 2]) * inv_mu};
    const double s_step[3] = {
        spin_values[offset],
        spin_values[offset + 1],
        spin_values[offset + 2]};

    double omega[3];
    jams::solvers::rkmk::omega_llg(
        s_step, h, gyro_eff_values[i], alpha_values[i], omega);
    const double v3[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const double phi[3] = {
        0.5 * k2_values[offset],
        0.5 * k2_values[offset + 1],
        0.5 * k2_values[offset + 2]};

    double k3_i[3];
    jams::solvers::rkmk::dexp_inv_so3(phi, v3, k3_i);
    k3_values[offset] = k3_i[0];
    k3_values[offset + 1] = k3_i[1];
    k3_values[offset + 2] = k3_i[2];

    const double s0[3] = {
        s_init_values[offset],
        s_init_values[offset + 1],
        s_init_values[offset + 2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(k3_i, s0, s_out);

    spin_values[offset] = s_out[0];
    spin_values[offset + 1] = s_out[1];
    spin_values[offset + 2] = s_out[2];
  }
}

void rkmk4_step_4(const double dt,
                  const double noise_dt,
                  const jams::MultiArray<double, 2>& s_init,
                  const jams::MultiArray<double, 2>& k1,
                  const jams::MultiArray<double, 2>& k2,
                  const jams::MultiArray<double, 2>& k3,
                  const jams::MultiArray<jams::Real, 1>& gyro_eff,
                  const jams::Real* const noise_values) {
  const auto field_view = globals::h.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto alpha_view = globals::alpha.host_view();
  const auto gyro_eff_view = gyro_eff.host_view();
  const auto s_init_view = s_init.host_view();
  const auto k1_view = k1.host_view();
  const auto k2_view = k2.host_view();
  const auto k3_view = k3.host_view();
  auto spin_view = globals::s.mutable_host_view();

  const auto* field_values = field_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* alpha_values = alpha_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  const auto* s_init_values = s_init_view.data();
  const auto* k1_values = k1_view.data();
  const auto* k2_values = k2_view.data();
  const auto* k3_values = k3_view.data();
  auto* spin_values = spin_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < globals::num_spins; ++i) {
    const int offset = 3 * i;
    const double inv_mu = static_cast<double>(inv_mu_values[i]);
    const double h[3] = {
        static_cast<double>(field_values[offset]) * inv_mu,
        static_cast<double>(field_values[offset + 1]) * inv_mu,
        static_cast<double>(field_values[offset + 2]) * inv_mu};
    const double s_step[3] = {
        spin_values[offset],
        spin_values[offset + 1],
        spin_values[offset + 2]};

    double omega[3];
    jams::solvers::rkmk::omega_llg(
        s_step, h, gyro_eff_values[i], alpha_values[i], omega);
    const double v4[3] = {dt * omega[0], dt * omega[1], dt * omega[2]};
    const double phi[3] = {
        k3_values[offset],
        k3_values[offset + 1],
        k3_values[offset + 2]};

    double k4[3];
    jams::solvers::rkmk::dexp_inv_so3(phi, v4, k4);

    const double k[3] = {
        (k1_values[offset] + 2.0 * k2_values[offset] +
         2.0 * k3_values[offset] + k4[0]) /
            6.0,
        (k1_values[offset + 1] + 2.0 * k2_values[offset + 1] +
         2.0 * k3_values[offset + 1] + k4[1]) /
            6.0,
        (k1_values[offset + 2] + 2.0 * k2_values[offset + 2] +
         2.0 * k3_values[offset + 2] + k4[2]) /
            6.0};

    const double s0[3] = {
        s_init_values[offset],
        s_init_values[offset + 1],
        s_init_values[offset + 2]};
    double s_out[3];
    jams::solvers::rkmk::rodrigues_rotate(k, s0, s_out);

    const jams::Real h_noise[3] = {
        noise_values[offset],
        noise_values[offset + 1],
        noise_values[offset + 2]};
    double s_noisy[3];
    jams::solvers::rkmk::noise_step_rodrigues(
        s_out,
        h_noise,
        gyro_eff_values[i],
        alpha_values[i],
        noise_dt,
        s_noisy);

    spin_values[offset] = s_noisy[0];
    spin_values[offset + 1] = s_noisy[1];
    spin_values[offset + 2] = s_noisy[2];
  }
}

}  // namespace

void RKMK2LLGSolver::initialize(const libconfig::Setting& settings) {
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  const auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  const auto t_min =
      jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  std::cout << "\ntimestep (ps) " << step_size_ << "\n";
  std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  initialize_gyro_eff(settings, gyro_eff_);

  const std::string thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", "langevin-white-cpu");
  register_thermostat(Thermostat::create(thermostat_name, 0.5 * time_step()));
  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  s_init_.resize(globals::num_spins, 3);
  phi_.resize(globals::num_spins, 3);
}

bool RKMK2LLGSolver::supports_persistent_parallel_step() const {
#if HAS_OMP
  return thermostat_ && thermostat_->supports_update_in_parallel()
      && supports_compute_fields_in_parallel();
#else
  return false;
#endif
}

void RKMK2LLGSolver::run_persistent_parallel_step() {
#if HAS_OMP
  const double t0 = time_;
  const double half_dt = 0.5 * step_size_;

#pragma omp parallel
  {
    const jams::Real* noise_values = nullptr;

    thermostat_->update_in_parallel();
#pragma omp single copyprivate(noise_values)
    {
      noise_values = thermostat_->data();
    }
    apply_noise_step_rodrigues(half_dt, gyro_eff_, noise_values, true);

    compute_fields_in_parallel();
    rkmk2_step_1(step_size_, s_init_, phi_, gyro_eff_, true);

#pragma omp single
    {
      time_ = t0 + half_dt;
    }

    compute_fields_in_parallel();

    thermostat_->update_in_parallel();
#pragma omp single copyprivate(noise_values)
    {
      noise_values = thermostat_->data();
    }
    rkmk2_step_2(step_size_, half_dt, s_init_, phi_, gyro_eff_, noise_values, true);

#pragma omp single
    {
      iteration_++;
      time_ = iteration_ * step_size_;
    }
  }
#else
  run();
#endif
}

void RKMK2LLGSolver::run() {
  if (supports_persistent_parallel_step()) {
    run_persistent_parallel_step();
    return;
  }

  const double t0 = time_;
  const double half_dt = 0.5 * step_size_;

  update_thermostat();
  apply_noise_step_rodrigues(half_dt, gyro_eff_, thermostat_->data());

  compute_fields();
  rkmk2_step_1(step_size_, s_init_, phi_, gyro_eff_);

  time_ = t0 + half_dt;

  compute_fields();

  update_thermostat();
  rkmk2_step_2(step_size_, half_dt, s_init_, phi_, gyro_eff_, thermostat_->data());

  iteration_++;
  time_ = iteration_ * step_size_;
}

void RKMK4LLGSolver::initialize(const libconfig::Setting& settings) {
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  const auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  const auto t_min =
      jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  std::cout << "\ntimestep (ps) " << step_size_ << "\n";
  std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  initialize_gyro_eff(settings, gyro_eff_);

  const std::string thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", "langevin-white-cpu");
  register_thermostat(Thermostat::create(thermostat_name, 0.5 * time_step()));
  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  s_init_.resize(globals::num_spins, 3);
  k1_.resize(globals::num_spins, 3);
  k2_.resize(globals::num_spins, 3);
  k3_.resize(globals::num_spins, 3);
}

void RKMK4LLGSolver::run() {
  const double t0 = time_;
  const double half_dt = 0.5 * step_size_;

  update_thermostat();
  apply_noise_step_rodrigues(half_dt, gyro_eff_, thermostat_->data());

  compute_fields();
  rkmk4_step_1(step_size_, s_init_, k1_, gyro_eff_);

  time_ = t0 + half_dt;

  compute_fields();
  rkmk4_step_2(step_size_, s_init_, k1_, k2_, gyro_eff_);

  compute_fields();
  rkmk4_step_3(step_size_, s_init_, k2_, k3_, gyro_eff_);

  time_ = t0 + step_size_;

  compute_fields();

  update_thermostat();
  rkmk4_step_4(
      step_size_, half_dt, s_init_, k1_, k2_, k3_, gyro_eff_, thermostat_->data());

  iteration_++;
  time_ = iteration_ * step_size_;
}
