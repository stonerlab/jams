//
// Created by Joseph Barker on 05/01/2026.
//

#include "jams/solvers/cuda_llg_rkmk4.h"

#include <iostream>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/helpers/defaults.h"
#include "jams/interface/config.h"
#include "jams/solvers/cuda_solver_functions.cuh"
#include "jams/solvers/llg_rkmk_thermal_kick.h"

#ifndef JAMS_DIAGNOSTIC_RKMK4_REALHI_FIELD_SCALE
#define JAMS_DIAGNOSTIC_RKMK4_REALHI_FIELD_SCALE 0
#endif

#ifndef JAMS_DIAGNOSTIC_RKMK4_EXPLICIT_SYNC
#define JAMS_DIAGNOSTIC_RKMK4_EXPLICIT_SYNC 0
#endif

#ifndef JAMS_DIAGNOSTIC_EXCHANGE_REALHI_ACCUMULATION
#define JAMS_DIAGNOSTIC_EXCHANGE_REALHI_ACCUMULATION 0
#endif

#ifndef JAMS_DIAGNOSTIC_REPORT_BUILD
#define JAMS_DIAGNOSTIC_REPORT_BUILD 0
#endif

namespace {

using jams::solvers::rkmk::ThermalKick;
using jams::solvers::rkmk::make_thermal_kick;

inline void rkmk4_diagnostic_sync(const cudaStream_t stream) {
#if JAMS_DIAGNOSTIC_RKMK4_EXPLICIT_SYNC
  CHECK_CUDA_STATUS(cudaStreamSynchronize(stream));
#else
  (void)stream;
#endif
}

#if JAMS_DIAGNOSTIC_RKMK4_REALHI_FIELD_SCALE
using Rkmk4FieldReal = jams::RealHi;

__device__ __forceinline__ Rkmk4FieldReal rkmk4_scale_field(const CudaUniformFieldScale field_scale,
                                                            const jams::Real field,
                                                            const unsigned) {
  return static_cast<jams::RealHi>(field) * static_cast<jams::RealHi>(field_scale.inv_mus);
}

__device__ __forceinline__ Rkmk4FieldReal rkmk4_scale_field(const CudaPerSpinFieldScale field_scale,
                                                            const jams::Real field,
                                                            const unsigned spin_index) {
  return static_cast<jams::RealHi>(field) *
         static_cast<jams::RealHi>(field_scale.inv_mus[spin_index]);
}

template <typename GyroParam, typename AlphaParam>
__device__ __forceinline__ void rkmk4_omega_llg(const double s[3], const Rkmk4FieldReal h[3],
                                                const GyroParam gyro, const AlphaParam alpha,
                                                const unsigned spin_index, double omega[3]) {
  jams::solvers::rkmk::omega_llg(s, h, static_cast<jams::RealHi>(gyro.get(spin_index)),
                                 static_cast<jams::RealHi>(alpha.get(spin_index)), omega);
}
#else
using Rkmk4FieldReal = jams::Real;

template <typename FieldScaleParam>
__device__ __forceinline__ Rkmk4FieldReal rkmk4_scale_field(const FieldScaleParam field_scale,
                                                            const jams::Real field,
                                                            const unsigned spin_index) {
  return field_scale.scale(field, spin_index);
}

template <typename GyroParam, typename AlphaParam>
__device__ __forceinline__ void rkmk4_omega_llg(const double s[3], const Rkmk4FieldReal h[3],
                                                const GyroParam gyro, const AlphaParam alpha,
                                                const unsigned spin_index, double omega[3]) {
  omega_llg(s, h, gyro.get(spin_index), alpha.get(spin_index), omega);
}
#endif

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rkmk4_stage_1(const double* stage_spins, double* initial_spins, double* k1_dev,
                              double* output_spins, jams::Real* field_spin_cache,
                              const jams::Real* fields, const jams::Real* stage_noise,
                              const GyroParam gyro,
                              const FieldScaleParam field_scale, const AlphaParam alpha,
                              const unsigned num_spins, const double dt) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  Rkmk4FieldReal h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = rkmk4_scale_field(field_scale, fields[offset + n], spin_index);
    if (stage_noise) h[n] += stage_noise[offset + n];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = stage_spins[offset + n];
    initial_spins[offset + n] = s[n];
  }

  double omega[3];
  rkmk4_omega_llg(s, h, gyro, alpha, spin_index, omega);

  double k1[3];
  for (auto n = 0; n < 3; ++n) {
    k1[n] = omega[n] * dt;
    k1_dev[offset + n] = k1[n];
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = 0.5 * k1[n];
  }

  double s_out[3];
  rodrigues_rotate(phi, s, s_out);
  rkmk_store_spin_and_cache(output_spins, field_spin_cache, offset, s_out);
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rkmk4_stage_2(const double* initial_spins, const double* stage_spins,
                              const double* k1_dev, double* k2_dev, double* output_spins,
                              jams::Real* field_spin_cache, const jams::Real* fields,
                              const jams::Real* stage_noise,
                              const GyroParam gyro, const FieldScaleParam field_scale,
                              const AlphaParam alpha, const unsigned num_spins, const double dt) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  Rkmk4FieldReal h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = rkmk4_scale_field(field_scale, fields[offset + n], spin_index);
    if (stage_noise) h[n] += stage_noise[offset + n];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = stage_spins[offset + n];
  }

  double omega[3];
  rkmk4_omega_llg(s, h, gyro, alpha, spin_index, omega);

  double v2[3];
  for (auto n = 0; n < 3; ++n) {
    v2[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = 0.5 * k1_dev[offset + n];
  }

  double k2[3];
  dexp_inv_so3(phi, v2, k2);

  for (auto n = 0; n < 3; ++n) {
    k2_dev[offset + n] = k2[n];
  }

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = initial_spins[offset + n];
  }

  double phi2[3];
  for (auto n = 0; n < 3; ++n) {
    phi2[n] = 0.5 * k2[n];
  }

  double s_out[3];
  rodrigues_rotate(phi2, s_init, s_out);
  rkmk_store_spin_and_cache(output_spins, field_spin_cache, offset, s_out);
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rkmk4_stage_3(const double* initial_spins, const double* stage_spins,
                              const double* k2_dev, double* k3_dev, double* output_spins,
                              jams::Real* field_spin_cache, const jams::Real* fields,
                              const jams::Real* stage_noise,
                              const GyroParam gyro, const FieldScaleParam field_scale,
                              const AlphaParam alpha, const unsigned num_spins, const double dt) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  Rkmk4FieldReal h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = rkmk4_scale_field(field_scale, fields[offset + n], spin_index);
    if (stage_noise) h[n] += stage_noise[offset + n];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = stage_spins[offset + n];
  }

  double omega[3];
  rkmk4_omega_llg(s, h, gyro, alpha, spin_index, omega);

  double v3[3];
  for (auto n = 0; n < 3; ++n) {
    v3[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = 0.5 * k2_dev[offset + n];
  }

  double k3[3];
  dexp_inv_so3(phi, v3, k3);

  for (auto n = 0; n < 3; ++n) {
    k3_dev[offset + n] = k3[n];
  }

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = initial_spins[offset + n];
  }

  double s_out[3];
  rodrigues_rotate(k3, s_init, s_out);
  rkmk_store_spin_and_cache(output_spins, field_spin_cache, offset, s_out);
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rkmk4_stage_4(const double* initial_spins, const double* stage_spins,
                              const double* k1_dev, const double* k2_dev, const double* k3_dev,
                              double* output_spins, jams::Real* field_spin_cache,
                              const jams::Real* fields, const jams::Real* stage_noise,
                              const jams::Real* split_noise,
                              const ThermalKick kick, const GyroParam gyro,
                              const FieldScaleParam field_scale,
                              const AlphaParam alpha, const unsigned num_spins, const double dt,
                              const double noise_dt) {
  const unsigned int spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;

  const unsigned int offset = 3u * spin_index;

  Rkmk4FieldReal h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = rkmk4_scale_field(field_scale, fields[offset + n], spin_index);
    if (stage_noise) h[n] += stage_noise[offset + n];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = stage_spins[offset + n];
  }

  double omega[3];
  rkmk4_omega_llg(s, h, gyro, alpha, spin_index, omega);

  double v4[3];
  for (auto n = 0; n < 3; ++n) {
    v4[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = k3_dev[offset + n];
  }

  double k4[3];
  dexp_inv_so3(phi, v4, k4);

  double k[3];
  for (auto n = 0; n < 3; ++n) {
    k[n] = (k1_dev[offset + n] + 2.0 * k2_dev[offset + n] +
                    2.0 * k3_dev[offset + n] + k4[n]) /
                   6.0;
  }

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = initial_spins[offset + n];
  }

  double s_out[3];
  rodrigues_rotate(k, s_init, s_out);

  if (split_noise) {
    kick.apply(s_out, split_noise + offset, gyro.get(spin_index), alpha.get(spin_index),
               noise_dt, spin_index, s_out);
  }
  rkmk_store_spin_and_cache(output_spins, field_spin_cache, offset, s_out);
}

}  // namespace

void CUDALLGRKMK4Solver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);
  initialize_gyro_eff(settings, gyro_eff_);
  initialize_stage_thermostat(thermostat_name, jams::thermal_field_coupling(gyro_eff_),
                              2, 2, "split kicks");
#if JAMS_DIAGNOSTIC_REPORT_BUILD
  std::cout << "  diagnostic build flags:\n"
            << "    exchange_realhi_accumulation = " << JAMS_DIAGNOSTIC_EXCHANGE_REALHI_ACCUMULATION
            << "\n"
            << "    rkmk4_realhi_field_scale = " << JAMS_DIAGNOSTIC_RKMK4_REALHI_FIELD_SCALE << "\n"
            << "    rkmk4_explicit_sync = " << JAMS_DIAGNOSTIC_RKMK4_EXPLICIT_SYNC << "\n";
#endif
#if JAMS_DIAGNOSTIC_RKMK4_REALHI_FIELD_SCALE
  std::cout << "  RKMK4 diagnostic: RealHi deterministic field scaling\n";
#endif
#if JAMS_DIAGNOSTIC_RKMK4_EXPLICIT_SYNC
  std::cout << "  RKMK4 diagnostic: explicit synchronization after every stage\n";
#endif
  const auto gyro_eff_choice = cuda_spin_parameter_choice(gyro_eff_);
  gyro_eff_is_uniform_ = gyro_eff_choice.is_uniform;
  gyro_eff_uniform_value_ = gyro_eff_choice.uniform_value;
  const auto alpha_choice = cuda_spin_parameter_choice(globals::alpha);
  alpha_is_uniform_ = alpha_choice.is_uniform;
  alpha_uniform_value_ = alpha_choice.uniform_value;
  const auto mus_choice = cuda_field_scale_choice(globals::inv_mus);
  mus_is_uniform_ = mus_choice.is_uniform;
  mus_uniform_inv_value_ = mus_choice.uniform_inv_mus;

  spins_initial_.resize(globals::num_spins, 3);
  k1_.resize(globals::num_spins, 3);
  k2_.resize(globals::num_spins, 3);
  k3_.resize(globals::num_spins, 3);
}

void CUDALLGRKMK4Solver::run() {
  dispatch_cuda_spin_parameters(
      {gyro_eff_is_uniform_, gyro_eff_uniform_value_}, {alpha_is_uniform_, alpha_uniform_value_},
      gyro_eff_, globals::alpha, [this](const auto gyro, const auto alpha) {
        dispatch_cuda_field_scale({mus_is_uniform_, mus_uniform_inv_value_}, globals::inv_mus,
                                  [this, gyro, alpha](const auto field_scale) {
                                    run_with_parameters(gyro, alpha, field_scale);
                                  });
      });
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
void CUDALLGRKMK4Solver::run_with_parameters(const GyroParam gyro, const AlphaParam alpha,
                                             const FieldScaleParam field_scale) {
  const double step_start_time = time_;
  const double dt = step_size_;
  const double half_dt = 0.5 * dt;
  const auto stream = jams::instance().cuda_master_stream().get();

  const dim3 block_size = {256, 1, 1};
  const auto grid_size =
      cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  jams::Real* field_spin_cache = mutable_field_spin_cache_device_data();

  ThermalKick kick;
  if (uses_stage_thermostat()) {
    // The physics module can modify spins at the start of the step.
    record_spin_barrier_event();
  } else {
    update_thermostat_on_stream(stream);
    kick = make_thermal_kick(*thermostat_, globals::inv_mus, half_dt,
                            uses_automatic_thermostat_sampling(), true);
    cuda_rkmk_noise_step_cache_kernel<<<grid_size, block_size, 0, stream>>>(
        globals::s.mutable_device_data(), field_spin_cache, thermostat_->device_data(), gyro, alpha,
        globals::num_spins, half_dt, kick);
    DEBUG_CHECK_CUDA_ASYNC_STATUS
    thermostat_->record_consumed(stream);
    rkmk4_diagnostic_sync(stream);
    record_spin_and_field_cache_barrier_event();
  }

  compute_fields();

  const auto* start_noise = sample_stage_thermostat_on_stream(0, stream);
  rkmk4_stage_1<<<grid_size, block_size, 0, stream>>>(
      globals::s.device_data(), spins_initial_.mutable_device_data(), k1_.mutable_device_data(),
      globals::s.mutable_device_data(), field_spin_cache, globals::h.device_data(), start_noise,
      gyro, field_scale, alpha, globals::num_spins, dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  if (uses_stage_thermostat()) thermostat_->record_consumed(stream);
  rkmk4_diagnostic_sync(stream);
  record_spin_and_field_cache_barrier_event();

  time_ = step_start_time + half_dt;

  compute_fields();

  const auto* midpoint_noise = sample_stage_thermostat_on_stream(1, stream);
  rkmk4_stage_2<<<grid_size, block_size, 0, stream>>>(
      spins_initial_.device_data(), globals::s.mutable_device_data(), k1_.device_data(),
      k2_.mutable_device_data(), globals::s.mutable_device_data(), field_spin_cache,
      globals::h.device_data(), midpoint_noise, gyro, field_scale, alpha, globals::num_spins, dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  rkmk4_diagnostic_sync(stream);
  record_spin_and_field_cache_barrier_event();

  compute_fields();

  rkmk4_stage_3<<<grid_size, block_size, 0, stream>>>(
      spins_initial_.device_data(), globals::s.device_data(), k2_.device_data(),
      k3_.mutable_device_data(), globals::s.mutable_device_data(), field_spin_cache,
      globals::h.device_data(), midpoint_noise, gyro, field_scale, alpha, globals::num_spins, dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  if (uses_stage_thermostat()) thermostat_->record_consumed(stream);
  rkmk4_diagnostic_sync(stream);
  record_spin_and_field_cache_barrier_event();

  time_ = step_start_time + dt;

  compute_fields();

  const jams::Real* split_noise = nullptr;
  if (!uses_stage_thermostat()) {
    update_thermostat_on_stream(stream);
    split_noise = thermostat_->device_data();
  }

  const auto* end_noise = sample_stage_thermostat_on_stream(2, stream);
  rkmk4_stage_4<<<grid_size, block_size, 0, stream>>>(
      spins_initial_.device_data(), globals::s.mutable_device_data(), k1_.device_data(),
      k2_.device_data(), k3_.device_data(), globals::s.mutable_device_data(), field_spin_cache,
      globals::h.device_data(), end_noise, split_noise, kick, gyro, field_scale, alpha,
      globals::num_spins, dt, half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(stream);
  rkmk4_diagnostic_sync(stream);
  record_spin_and_field_cache_barrier_event();

  advance_time();
}
