// Copyright 2014 Joseph Barker. All rights reserved.

#include "cpu_llg_heun.h"

#include <cmath>
#include <jams/interface/config.h>
#include <jams/helpers/maths.h>

#include "jams/core/globals.h"
#include "jams/core/physics.h"

namespace {

inline jams::Vec<double, 3> llg_rhs(
    const double sx,
    const double sy,
    const double sz,
    const double hx,
    const double hy,
    const double hz,
    const double alpha,
    const double gyro_eff) {
  const double cx = sy * hz - sz * hy;
  const double cy = sz * hx - sx * hz;
  const double cz = sx * hy - sy * hx;

  const double scx = sy * cz - sz * cy;
  const double scy = sz * cx - sx * cz;
  const double scz = sx * cy - sy * cx;

  return {
      -gyro_eff * (cx + alpha * scx),
      -gyro_eff * (cy + alpha * scy),
      -gyro_eff * (cz + alpha * scz)};
}

inline jams::Vec<double, 3> normalized_spin(
    const double sx,
    const double sy,
    const double sz) {
  const double norm_sq = sx * sx + sy * sy + sz * sz;
  if (norm_sq == 0.0) {
    return {0.0, 0.0, 0.0};
  }
  const double inv_norm = 1.0 / std::sqrt(norm_sq);
  return {sx * inv_norm, sy * inv_norm, sz * inv_norm};
}

}  // namespace

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
  {
    const auto spin_view = globals::s.host_view();
    auto old_spin_view = s_old_.mutable_host_view();
    const auto* spin_values = spin_view.data();
    auto* old_spin_values = old_spin_view.data();
#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto n = 0; n < globals::num_spins3; ++n) {
      old_spin_values[n] = spin_values[n];
    }
  }

  update_thermostat();

  Solver::compute_fields();

  const auto alpha_view = globals::alpha.host_view();
  const auto inv_mu_view = globals::inv_mus.host_view();
  const auto gyro_eff_view = gyro_eff_.host_view();
  const auto* alpha_values = alpha_view.data();
  const auto* inv_mu_values = inv_mu_view.data();
  const auto* gyro_eff_values = gyro_eff_view.data();
  const auto* thermostat_values = thermostat_->data();

  {
    auto spin_view = globals::s.mutable_host_view();
    auto field_view = globals::h.mutable_host_view();
    auto ds_dt_view = globals::ds_dt.mutable_host_view();
    auto* spin_values = spin_view.data();
    auto* field_values = field_view.data();
    auto* ds_dt_values = ds_dt_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto offset = 3 * i;
      const double inv_mu = static_cast<double>(inv_mu_values[i]);
      const double hx = static_cast<double>(thermostat_values[offset] + field_values[offset] * inv_mu);
      const double hy = static_cast<double>(thermostat_values[offset + 1] + field_values[offset + 1] * inv_mu);
      const double hz = static_cast<double>(thermostat_values[offset + 2] + field_values[offset + 2] * inv_mu);

      field_values[offset] = static_cast<jams::Real>(hx);
      field_values[offset + 1] = static_cast<jams::Real>(hy);
      field_values[offset + 2] = static_cast<jams::Real>(hz);

      const double sx = spin_values[offset];
      const double sy = spin_values[offset + 1];
      const double sz = spin_values[offset + 2];
      const auto rhs = llg_rhs(
          sx,
          sy,
          sz,
          hx,
          hy,
          hz,
          static_cast<double>(alpha_values[i]),
          static_cast<double>(gyro_eff_values[i]));

      ds_dt_values[offset] = 0.5 * rhs[0];
      ds_dt_values[offset + 1] = 0.5 * rhs[1];
      ds_dt_values[offset + 2] = 0.5 * rhs[2];

      const auto spin = normalized_spin(
          sx + step_size_ * rhs[0],
          sy + step_size_ * rhs[1],
          sz + step_size_ * rhs[2]);

      spin_values[offset] = spin[0];
      spin_values[offset + 1] = spin[1];
      spin_values[offset + 2] = spin[2];
    }
  }

  double mid_time_step = step_size_;
  time_ = t0 + mid_time_step;

  Solver::compute_fields();

  {
    auto spin_view = globals::s.mutable_host_view();
    const auto old_spin_view = s_old_.host_view();
    auto field_view = globals::h.mutable_host_view();
    auto ds_dt_view = globals::ds_dt.mutable_host_view();
    auto* spin_values = spin_view.data();
    const auto* old_spin_values = old_spin_view.data();
    auto* field_values = field_view.data();
    auto* ds_dt_values = ds_dt_view.data();

#if HAS_OMP
#pragma omp parallel for schedule(static)
#endif
    for (auto i = 0; i < globals::num_spins; ++i) {
      const auto offset = 3 * i;
      const double inv_mu = static_cast<double>(inv_mu_values[i]);
      const double hx = static_cast<double>(thermostat_values[offset] + field_values[offset] * inv_mu);
      const double hy = static_cast<double>(thermostat_values[offset + 1] + field_values[offset + 1] * inv_mu);
      const double hz = static_cast<double>(thermostat_values[offset + 2] + field_values[offset + 2] * inv_mu);

      field_values[offset] = static_cast<jams::Real>(hx);
      field_values[offset + 1] = static_cast<jams::Real>(hy);
      field_values[offset + 2] = static_cast<jams::Real>(hz);

      const double sx = spin_values[offset];
      const double sy = spin_values[offset + 1];
      const double sz = spin_values[offset + 2];
      const auto rhs = llg_rhs(
          sx,
          sy,
          sz,
          hx,
          hy,
          hz,
          static_cast<double>(alpha_values[i]),
          static_cast<double>(gyro_eff_values[i]));

      const double dsx = ds_dt_values[offset] + 0.5 * rhs[0];
      const double dsy = ds_dt_values[offset + 1] + 0.5 * rhs[1];
      const double dsz = ds_dt_values[offset + 2] + 0.5 * rhs[2];

      ds_dt_values[offset] = dsx;
      ds_dt_values[offset + 1] = dsy;
      ds_dt_values[offset + 2] = dsz;

      const auto spin = normalized_spin(
          old_spin_values[offset] + step_size_ * dsx,
          old_spin_values[offset + 1] + step_size_ * dsy,
          old_spin_values[offset + 2] + step_size_ * dsz);

      spin_values[offset] = spin[0];
      spin_values[offset + 1] = spin[1];
      spin_values[offset + 2] = spin[2];
    }
  }

  iteration_++;
  time_ = iteration_ * step_size_;
}
