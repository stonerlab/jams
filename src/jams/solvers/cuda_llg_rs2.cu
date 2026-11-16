#include "jams/solvers/cuda_llg_rs2.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/core/physics.h"
#include "jams/helpers/defaults.h"
#include "jams/interface/config.h"
#include "jams/solvers/cuda_solver_functions.cuh"
#include "jams/thermostats/cuda_thermostat_classical.h"

namespace {

template <typename FieldScaleParam>
__device__ void rs2_drift(const double spin[3], const double u[3],
                          const jams::Real* fields, const jams::Real* noise,
                          const FieldScaleParam field_scale, const unsigned spin_index,
                          const jams::Real gyro, const jams::Real alpha, double f[3]) {
  const auto offset = 3u * spin_index;
  jams::Real field[3];
  for (auto n = 0; n < 3; ++n) {
    field[n] = fields ? field_scale.scale(fields[offset + n], spin_index) : jams::Real{0};
    if (noise) field[n] += noise[offset + n];
  }

  double omega[3];
  omega_llg(spin, field, gyro, alpha, omega);
  dexp_inv_so3(u, omega, f);
}

__device__ void rs2_diffusion(const double spin_initial[3], const double u[3],
                              const int axis, const double sigma, const jams::Real gyro,
                              const jams::Real alpha, double g[3]) {
  double field[3] = {};
  field[axis] = sigma;
  double spin[3];
  double omega[3];
  rodrigues_rotate(u, spin_initial, spin);
  jams::solvers::rkmk::omega_llg(spin, field, gyro, alpha, omega);
  dexp_inv_so3(u, omega, g);
}

__device__ void rs2_diffusion_stages_1_and_2(const double spin_initial[3],
                                           const double* draws, const double x_amplitude,
                                           const double sigma, const jams::Real gyro,
                                           const jams::Real alpha, double x[3],
                                           double g1[3][3], double g2[3][3]) {
  const double u_zero[3] = {};
  for (int k = 0; k < 3; ++k) {
    x[k] = draws[k] <= 1.0 / 6.0 ? -x_amplitude
        : draws[k] > 5.0 / 6.0 ? x_amplitude : 0.0;
    rs2_diffusion(spin_initial, u_zero, k, sigma, gyro, alpha, g1[k]);
    double u[3];
    for (auto n = 0; n < 3; ++n) {
      u[n] = (2.0 / 3.0) * x[k] * g1[k][n];
    }
    rs2_diffusion(spin_initial, u, k, sigma, gyro, alpha, g2[k]);
  }
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rs2_stage_1(double* spins, double* initial_spins, double* f1_dev,
                             jams::Real* field_spin_cache, const jams::Real* fields,
                             const jams::Real* noise, const GyroParam gyro,
                             const FieldScaleParam field_scale, const AlphaParam alpha,
                             const unsigned num_spins, const double dt) {
  const unsigned spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;
  const unsigned offset = 3u * spin_index;

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = spins[offset + n];
    initial_spins[offset + n] = s[n];
  }

  const double u_zero[3] = {};
  double f1[3];
  rs2_drift(s, u_zero, fields, noise, field_scale, spin_index,
             gyro.get(spin_index), alpha.get(spin_index), f1);

  double u2[3];
  for (auto n = 0; n < 3; ++n) {
    f1_dev[offset + n] = f1[n];
    u2[n] = (2.0 / 3.0) * dt * f1[n];
  }
  double s_out[3];
  rodrigues_rotate(u2, s, s_out);
  rkmk_store_spin_and_cache(spins, field_spin_cache, offset, s_out);
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rs2_stage_2(const double* initial_spins, const double* f1_dev, double* f2_dev,
                             double* b_dev, double* spins, jams::Real* field_spin_cache,
                             const jams::Real* fields, const jams::Real* noise,
                             const double* sigma, const double* uniforms, const GyroParam gyro,
                             const FieldScaleParam field_scale, const AlphaParam alpha,
                             const unsigned num_spins, const double dt, const double x_amplitude) {
  const unsigned spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;
  const unsigned offset = 3u * spin_index;
  const auto gyro_i = gyro.get(spin_index);
  const auto alpha_i = alpha.get(spin_index);

  double s[3];
  double s_initial[3];
  double u2[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = spins[offset + n];
    s_initial[n] = initial_spins[offset + n];
    u2[n] = (2.0 / 3.0) * dt * f1_dev[offset + n];
  }
  double f2[3];
  rs2_drift(s, u2, fields, noise, field_scale, spin_index, gyro_i, alpha_i, f2);

  double b[3] = {};
  if (sigma && sigma[spin_index] != 0.0) {
    double x[3];
    double g1[3][3];
    double g2[3][3];
    rs2_diffusion_stages_1_and_2(s_initial, uniforms + 5ull * spin_index, x_amplitude,
                                 sigma[spin_index], gyro_i, alpha_i, x, g1, g2);
    for (int k = 0; k < 3; ++k) {
      for (auto n = 0; n < 3; ++n) {
        b[n] += x[k] * (0.25 * g1[k][n] + 0.75 * g2[k][n]);
      }
    }
  }

  double u3[3];
  for (auto n = 0; n < 3; ++n) {
    f2_dev[offset + n] = f2[n];
    b_dev[offset + n] = b[n];
    u3[n] = dt * (f1_dev[offset + n] / 6.0 + 0.5 * f2[n]) + b[n];
  }
  double s_out[3];
  rodrigues_rotate(u3, s_initial, s_out);
  rkmk_store_spin_and_cache(spins, field_spin_cache, offset, s_out);
}

template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
__global__ void rs2_stage_3(const double* initial_spins, const double* f1_dev, const double* f2_dev,
                             const double* b_dev, double* spins, jams::Real* field_spin_cache,
                             const jams::Real* fields, const jams::Real* noise,
                             const double* sigma, const double* uniforms, const GyroParam gyro,
                             const FieldScaleParam field_scale, const AlphaParam alpha,
                             const unsigned num_spins, const double dt, const double sqrt_dt,
                             const double x_amplitude) {
  const unsigned spin_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (spin_index >= num_spins) return;
  const unsigned offset = 3u * spin_index;
  const auto gyro_i = gyro.get(spin_index);
  const auto alpha_i = alpha.get(spin_index);

  double s[3];
  double s_initial[3];
  double u3[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = spins[offset + n];
    s_initial[n] = initial_spins[offset + n];
    u3[n] = dt * (f1_dev[offset + n] / 6.0 + 0.5 * f2_dev[offset + n]) + b_dev[offset + n];
  }
  double f3[3];
  rs2_drift(s, u3, fields, noise, field_scale, spin_index, gyro_i, alpha_i, f3);

  double u_out[3];
  for (auto n = 0; n < 3; ++n) {
    u_out[n] = dt * (0.25 * f1_dev[offset + n] + 0.25 * f2_dev[offset + n] + 0.5 * f3[n]);
  }

  if (sigma && sigma[spin_index] != 0.0) {
    const auto sigma_i = sigma[spin_index];
    const auto* draws = uniforms + 5ull * spin_index;
    const double z[2] = {draws[3] <= 0.5 ? -1.0 : 1.0, draws[4] <= 0.5 ? -1.0 : 1.0};
    double x[3];
    double g1[3][3];
    double g2[3][3];
    rs2_diffusion_stages_1_and_2(s_initial, draws, x_amplitude, sigma_i, gyro_i, alpha_i, x, g1, g2);

    for (int k = 0; k < 3; ++k) {
      double u[3];
      double g3[3];
      double g4[3];
      for (auto n = 0; n < 3; ++n) {
        u[n] = dt * f1_dev[offset + n] + b_dev[offset + n]
            - x[k] * (g1[k][n] / 6.0 + 0.5 * g2[k][n]);
      }
      rs2_diffusion(s_initial, u, k, sigma_i, gyro_i, alpha_i, g3);
      for (auto n = 0; n < 3; ++n) {
        u[n] = dt * f1_dev[offset + n] + b_dev[offset + n]
            + x[k] * (-1.5 * g1[k][n] - 0.5 * g2[k][n] + 2.0 * g3[n]);
      }
      rs2_diffusion(s_initial, u, k, sigma_i, gyro_i, alpha_i, g4);

      // z stores Z/sqrt(dt), so only the sign of the lower axis is needed.
      double p[3] = {};
      for (int l = 0; l < 3; ++l) {
        if (l == k) continue;
        const double v = k > l ? x[k] * z[l] : -x[l] * z[k];
        for (auto n = 0; n < 3; ++n) {
          p[n] += v * g1[l][n];
        }
      }

      const double minus_p[3] = {-p[0], -p[1], -p[2]};
      double g_minus[3];
      double g_plus[3];
      rs2_diffusion(s_initial, minus_p, k, sigma_i, gyro_i, alpha_i, g_minus);
      rs2_diffusion(s_initial, p, k, sigma_i, gyro_i, alpha_i, g_plus);

      for (auto n = 0; n < 3; ++n) {
        u_out[n] += (x[k] / 8.0) * (g1[k][n] + 3.0 * g2[k][n] + 3.0 * g3[n] + g4[n])
            + (sqrt_dt / 4.0) * (g_minus[n] - g_plus[n]);
      }
    }
  }

  double s_out[3];
  rodrigues_rotate(u_out, s_initial, s_out);
  rkmk_store_spin_and_cache(spins, field_spin_cache, offset, s_out);
}

}  // namespace

void CUDALLGRS2Solver::initialize(const libconfig::Setting& settings) {
  initialize_time(settings);

  const auto thermostat_name = jams::config_optional<std::string>(
      settings, "thermostat", jams::defaults::solver_gpu_thermostat);
  const auto num_spins = globals::num_spins;

  initialize_gyro_eff(settings, gyro_eff_);
  register_thermostat(Thermostat::create(
      thermostat_name, time_step(), jams::thermal_field_coupling(gyro_eff_), time_step() / 3.0));
  const auto* white_bath = dynamic_cast<const CudaThermostatClassical*>(thermostat_.get());
  has_coloured_bath_ = white_bath == nullptr;
  coloured_sample_ready_ = false;
  white_generator_.reset();

  const auto gyro_choice = cuda_spin_parameter_choice(gyro_eff_);
  gyro_eff_is_uniform_ = gyro_choice.is_uniform;
  gyro_eff_uniform_value_ = gyro_choice.uniform_value;
  const auto alpha_choice = cuda_spin_parameter_choice(globals::alpha);
  alpha_is_uniform_ = alpha_choice.is_uniform;
  alpha_uniform_value_ = alpha_choice.uniform_value;
  const auto mus_choice = cuda_field_scale_choice(globals::inv_mus);
  mus_is_uniform_ = mus_choice.is_uniform;
  mus_uniform_inv_value_ = mus_choice.uniform_inv_mus;

  spins_initial_.resize(num_spins, 3);
  f1_.resize(num_spins, 3);
  f2_.resize(num_spins, 3);
  b_.resize(num_spins, 3);

  if (!white_bath) return;
  white_temperature_ = jams::config_required<double>(globals::config->lookup("physics"), "temperature");
  white_sigma_ = white_bath->diffusion_amplitudes(white_temperature_);
  const auto amplitudes = white_sigma_.host_span();

  if (std::any_of(amplitudes.begin(), amplitudes.end(), [](double sigma) { return sigma > 0.0; })) {
    uniforms_.resize(num_spins, 5);
    curandGenerator_t generator = nullptr;
    CHECK_CURAND_STATUS(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    white_generator_.reset(generator);
    CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(generator, jams::instance().random_generator()()));
    CHECK_CURAND_STATUS(curandSetStream(generator, jams::instance().cuda_master_stream().get()));
  }
}

void CUDALLGRS2Solver::run() {
  if (!has_coloured_bath_) {
    if (thermostat_->has_uniform_temperature() && physics_module_->temperature() != white_temperature_) {
      throw jams::ConfigException(globals::config->lookup("physics"),
                                  "llg-rs2-gpu requires a fixed white-bath temperature");
    }
    if (white_generator_) {
      CHECK_CURAND_STATUS(curandGenerateUniformDouble(
          white_generator_.get(), uniforms_.mutable_device_data(), uniforms_.size()));
    }
  }
  run_step();
}

void CUDALLGRS2Solver::run_step() {
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
void CUDALLGRS2Solver::run_with_parameters(const GyroParam gyro, const AlphaParam alpha,
                                          const FieldScaleParam field_scale) {
  const double dt = time_step();
  const double sqrt_dt = std::sqrt(dt);
  const double x_amplitude = std::sqrt(3.0) * sqrt_dt;
  const double stage_time = time_ + (2.0 / 3.0) * dt;
  const auto stream = jams::instance().cuda_master_stream().get();
  const auto num_spins = static_cast<unsigned>(globals::num_spins);
  const dim3 block_size = {256, 1, 1};
  const auto grid_size = cuda_grid_size(block_size, {num_spins, 1, 1});

  if (has_coloured_bath_ && !coloured_sample_ready_) {
    thermostat_->wait_for_consumed();
    if (thermostat_->has_uniform_temperature()) {
      thermostat_->set_temperature(physics_module_->temperature());
    }
    thermostat_->sample_initial();
    thermostat_->record_done();
    thermostat_->wait_on(stream);
    coloured_sample_ready_ = true;
  }

  const auto* noise = has_coloured_bath_ ? thermostat_->device_data() : nullptr;
  const auto* sigma = !has_coloured_bath_ && white_generator_ ? white_sigma_.device_data() : nullptr;
  const auto* uniforms = sigma ? uniforms_.device_data() : nullptr;
  const auto stage_fields = [this]() {
    return hamiltonians_.empty() ? nullptr : globals::h.device_data();
  };
  auto* spins = globals::s.mutable_device_data();
  auto* field_spin_cache = mutable_field_spin_cache_device_data();

  // Physics modules can change spins between completed steps.
  record_spin_barrier_event();
  compute_fields();
  rs2_stage_1<<<grid_size, block_size, 0, stream>>>(
      spins, spins_initial_.mutable_device_data(), f1_.mutable_device_data(), field_spin_cache,
      stage_fields(), noise, gyro, field_scale, alpha, num_spins, dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_and_field_cache_barrier_event();

  if (has_coloured_bath_) {
    thermostat_->record_consumed(stream);
    update_thermostat();                  // t_n + dt/3
    update_thermostat_on_stream(stream);  // t_n + 2*dt/3
  }

  time_ = stage_time;
  compute_fields();
  rs2_stage_2<<<grid_size, block_size, 0, stream>>>(
      spins_initial_.device_data(), f1_.device_data(), f2_.mutable_device_data(), b_.mutable_device_data(),
      spins, field_spin_cache, stage_fields(), noise, sigma, uniforms, gyro, field_scale, alpha,
      num_spins, dt, x_amplitude);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_and_field_cache_barrier_event();

  compute_fields();
  rs2_stage_3<<<grid_size, block_size, 0, stream>>>(
      spins_initial_.device_data(), f1_.device_data(), f2_.device_data(), b_.device_data(),
      spins, field_spin_cache, stage_fields(), noise, sigma, uniforms, gyro, field_scale, alpha,
      num_spins, dt, sqrt_dt, x_amplitude);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_and_field_cache_barrier_event();

  if (has_coloured_bath_) {
    thermostat_->record_consumed(stream);
    update_thermostat_on_stream(stream);  // t_n + dt, retained for the next step
  }
  advance_time();
}
