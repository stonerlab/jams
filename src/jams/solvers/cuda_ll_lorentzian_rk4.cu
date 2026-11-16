// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/solvers/cuda_ll_lorentzian_rk4.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/cuda/cuda_common.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/defaults.h"
#include "jams/helpers/maths.h"
#include "jams/interface/config.h"
#include "jams/solvers/cuda_rk4_base_kernel.cuh"

namespace {

__global__ void cuda_ll_lorentzian_rk4_kernel(
    double* s_k_dev, double* w_k_dev, double* v_k_dev, const double* spins, const double* w_dev,
    const double* v_dev, const jams::Real* fields, const jams::Real* noise,
    const jams::Real* gyro_eff, const jams::Real* mus_dev, const jams::Real* inverse_moments,
    const double lorentzian_omega, const double lorentzian_gamma, const double lorentzian_A,
    const unsigned num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (spin_index >= num_spins) return;
  const unsigned int offset = 3 * spin_index;

  double v[3];
  for (auto n = 0; n < 3; ++n) {
    v[n] = v_dev[offset + n];
  }

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = ((fields[offset + n] * inverse_moments[spin_index]) +
                    noise[offset + n] + v[n]);
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = spins[offset + n];
  }

  double w[3];
  for (auto n = 0; n < 3; ++n) {
    w[n] = w_dev[offset + n];
  }

  double sxh[3] = {(s[1] * h[2] - s[2] * h[1]), (s[2] * h[0] - s[0] * h[2]),
                   (s[0] * h[1] - s[1] * h[0])};

  for (auto n = 0; n < 3; ++n) {
    s_k_dev[offset + n] = -gyro_eff[spin_index] * sxh[n];
  }

  // TECHNICAL NOTE
  // --------------
  // The \vec{S} in equations in Anders arXiv:2009.00600v2 are dimensionful
  // in units of \hbar. So if the quantum spin is 1/2  then
  // |\vec{S}| = \hbar / 2. In JAMS we use unit vectors for spins. Therefore
  // the differential equations of the memory kernel need rewriting.
  // The simplest way is to realise that \gamma\vec{S} = \mu_s \vec{e} where
  // \vec{e} is the unit vector of a spin and \mu_s is the classical moment
  // \mu_s = g S \mu_B. Hence we can write:
  //
  // dW(t)/dt = -\omega_0^2 V(t) - \Gamma W(t) + A \mu_s e(t)
  //
  // Noting also that in Anders conventions the gyromagnetic ratio gamma is
  // negative i.e. their spins point in the opposite direction to their
  // magnetic moments (technically correct), but after substituting
  // \gamma\vec{S} = \mu_s \vec{e} we retain the positive sign on the final
  // term of dW/dt.
  //
  for (auto n = 0; n < 3; ++n) {
    w_k_dev[offset + n] = -lorentzian_omega * lorentzian_omega * v[n] -
                                  lorentzian_gamma * w[n] +
                                  lorentzian_A * mus_dev[spin_index] * s[n];
  }

  for (auto n = 0; n < 3; ++n) {
    v_k_dev[offset + n] = w[n];
  }
}

__global__ void combine_and_normalise_spins(double* spins, const double* initial_spins,
                                            const double* k1, const double* k2, const double* k3,
                                            const double* k4, const double dt,
                                            const unsigned int num_spins) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;
  const unsigned int offset = 3 * spin_index;

  double spin[3];
  for (int n = 0; n < 3; ++n) {
    const unsigned int index = offset + n;
    spin[n] =
        initial_spins[index] + dt * (k1[index] + 2 * k2[index] + 2 * k3[index] + k4[index]) / 6.0;
  }
  const double norm_sq = norm_squared(spin);
  const double inv_norm = norm_sq > 0.0 ? rsqrt(norm_sq) : 0.0;
  for (int n = 0; n < 3; ++n) {
    spins[offset + n] = spin[n] * inv_norm;
  }
}

}  // namespace

void CUDALLLorentzianRK4Solver::initialize(const libconfig::Setting& settings) {
  if (globals::lattice->num_materials() > 1) {
    throw std::runtime_error(
        "CUDALLLorentzianRK4Solver is only implemented for single material cells");
  }

  initialize_time(settings);

  lorentzian_gamma_ = kTwoPi * jams::config_required<double>(globals::config->lookup("thermostat"),
                                                             "lorentzian_gamma");
  lorentzian_omega_ = kTwoPi * jams::config_required<double>(globals::config->lookup("thermostat"),
                                                             "lorentzian_omega0");

  // In arXiv:2009.00600v2 Janet uses eta_G for the Gilbert damping, but this is
  // a **dimensionful** Gilbert damping (implied by Eq. (1) in the paper and
  // also explicitly mentioned). In JAMS alpha is the dimensionless Gilbert
  // damping. The difference is eta_G = alpha / (mu_s * gamma). It's important
  // that we convert here to get the scaling of the noise correct (i.e. it
  // converts Janet's equations into the JAMS convention).

  const int first_spin = globals::first_magnetic_spin();
  double eta_G =
      globals::alpha(first_spin) * globals::inv_mus(first_spin) / globals::gyro(first_spin);

  lorentzian_A_ = (eta_G * pow4(lorentzian_omega_)) / (lorentzian_gamma_);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name, time_step()));

  std::cout << "  thermostat " << thermostat_name << "\n";

  initialize_gyro_eff(settings, gyro_eff_);

  spins_initial_.resize(globals::num_spins, 3);

  s_k1_.resize(globals::num_spins, 3);
  s_k2_.resize(globals::num_spins, 3);
  s_k3_.resize(globals::num_spins, 3);
  s_k4_.resize(globals::num_spins, 3);

  zero(w_memory_process_.resize(globals::num_spins, 3));
  w_memory_initial_.resize(globals::num_spins, 3);
  w_memory_process_k1_.resize(globals::num_spins, 3);
  w_memory_process_k2_.resize(globals::num_spins, 3);
  w_memory_process_k3_.resize(globals::num_spins, 3);
  w_memory_process_k4_.resize(globals::num_spins, 3);

  zero(v_memory_process_.resize(globals::num_spins, 3));
  v_memory_initial_.resize(globals::num_spins, 3);
  v_memory_process_k1_.resize(globals::num_spins, 3);
  v_memory_process_k2_.resize(globals::num_spins, 3);
  v_memory_process_k3_.resize(globals::num_spins, 3);
  v_memory_process_k4_.resize(globals::num_spins, 3);
}

void CUDALLLorentzianRK4Solver::run() {
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;
  const auto stream = jams::instance().cuda_master_stream().get();

  const dim3 block_size = {64, 1, 1};
  const auto num_spins = static_cast<unsigned int>(globals::num_spins);
  const auto num_components = static_cast<unsigned int>(globals::num_spins3);
  const auto spin_grid_size = cuda_grid_size(block_size, {num_spins, 1, 1});
  const auto component_grid_size = cuda_grid_size(block_size, {num_components, 1, 1});

  const auto snapshot = [stream](auto& initial, const auto& state) {
    CHECK_CUDA_STATUS(cudaMemcpyAsync(initial.mutable_device_data(), state.device_data(),
                                      state.bytes(), cudaMemcpyDeviceToDevice, stream));
  };
  snapshot(spins_initial_, globals::s);
  snapshot(w_memory_initial_, w_memory_process_);
  snapshot(v_memory_initial_, v_memory_process_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();
  // Some thermostats also enqueue work on auxiliary streams.
  CHECK_CUDA_STATUS(cudaDeviceSynchronize());
  const auto* noise = thermostat_->device_data();

  const auto evaluate_stage = [&](auto& spin_slope, auto& w_slope, auto& v_slope) {
    compute_fields();
    cuda_ll_lorentzian_rk4_kernel<<<spin_grid_size, block_size, 0, stream>>>(
        spin_slope.mutable_device_data(), w_slope.mutable_device_data(),
        v_slope.mutable_device_data(), globals::s.device_data(), w_memory_process_.device_data(),
        v_memory_process_.device_data(), globals::h.device_data(), noise, gyro_eff_.device_data(),
        globals::mus.device_data(), globals::inv_mus.device_data(), lorentzian_omega_,
        lorentzian_gamma_, lorentzian_A_, num_spins);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  };

  const auto prepare_stage = [&](const double stage_dt, const auto& spin_slope, const auto& w_slope,
                                 const auto& v_slope) {
    const auto update = [&](auto& state, const auto& initial, const auto& slope) {
      cuda_rk4_mid_step_kernel<<<component_grid_size, block_size, 0, stream>>>(
          num_components, stage_dt, initial.device_data(), slope.device_data(),
          state.mutable_device_data());
    };
    update(globals::s, spins_initial_, spin_slope);
    update(w_memory_process_, w_memory_initial_, w_slope);
    update(v_memory_process_, v_memory_initial_, v_slope);
    record_spin_barrier_event();
  };

  evaluate_stage(s_k1_, w_memory_process_k1_, v_memory_process_k1_);

  time_ = step_start_time + half_dt;
  prepare_stage(half_dt, s_k1_, w_memory_process_k1_, v_memory_process_k1_);
  evaluate_stage(s_k2_, w_memory_process_k2_, v_memory_process_k2_);

  prepare_stage(half_dt, s_k2_, w_memory_process_k2_, v_memory_process_k2_);
  evaluate_stage(s_k3_, w_memory_process_k3_, v_memory_process_k3_);

  time_ = step_start_time + dt;
  prepare_stage(dt, s_k3_, w_memory_process_k3_, v_memory_process_k3_);
  evaluate_stage(s_k4_, w_memory_process_k4_, v_memory_process_k4_);
  thermostat_->record_consumed(stream);

  combine_and_normalise_spins<<<spin_grid_size, block_size, 0, stream>>>(
      globals::s.mutable_device_data(), spins_initial_.device_data(), s_k1_.device_data(),
      s_k2_.device_data(), s_k3_.device_data(), s_k4_.device_data(), dt, num_spins);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  const auto combine = [&](auto& state, const auto& initial, const auto& k1, const auto& k2,
                           const auto& k3, const auto& k4) {
    cuda_rk4_combination_kernel<<<component_grid_size, block_size, 0, stream>>>(
        state.mutable_device_data(), initial.device_data(), k1.device_data(), k2.device_data(),
        k3.device_data(), k4.device_data(), dt, num_components);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  };
  combine(w_memory_process_, w_memory_initial_, w_memory_process_k1_, w_memory_process_k2_,
          w_memory_process_k3_, w_memory_process_k4_);
  combine(v_memory_process_, v_memory_initial_, v_memory_process_k1_, v_memory_process_k2_,
          v_memory_process_k3_, v_memory_process_k4_);

  record_spin_barrier_event();
  advance_time();
}
