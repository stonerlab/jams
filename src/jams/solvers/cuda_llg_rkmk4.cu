//
// Created by Joseph Barker on 05/01/2026.
//

#include "jams/solvers/cuda_llg_rkmk4.h"

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/solvers/cuda_solver_functions.cuh"

__global__ void cuda_llg_rkmk4_kernel_step_1
(
  const double * s_init_dev,
  double * k1_dev,
  double * s_step_dev,
  const jams::Real * h_step_dev,
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const unsigned dev_num_spins,
  const double dt
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned int base = 3u * idx;

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = h_step_dev[base + n] / mus_dev[idx];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = s_init_dev[base + n];
  }

  double omega[3];
  omega_llg(s, h, gyro_dev[idx], alpha_dev[idx], omega);

  double k1[3];
  for (auto n = 0; n < 3; ++n) {
    k1[n] = omega[n] * dt;
    k1_dev[base + n] = k1[n];
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = 0.5 * k1[n];
  }

  double s_out[3];
  rodrigues_rotate(phi, s, s_out);

  for (auto n = 0; n < 3; ++n) {
    s_step_dev[base + n] = s_out[n];
  }
}

__global__ void cuda_llg_rkmk4_kernel_step_2
(
  const double * s_init_dev,
  const double * s_step_dev,
  const double * k1_dev,
  double * k2_dev,
  double * s_out_dev,
  const jams::Real * h_step_dev,
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const unsigned dev_num_spins,
  const double dt
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned int base = 3u * idx;

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = h_step_dev[base + n] / mus_dev[idx];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = s_step_dev[base + n];
  }

  double omega[3];
  omega_llg(s, h, gyro_dev[idx], alpha_dev[idx], omega);

  double v2[3];
  for (auto n = 0; n < 3; ++n) {
    v2[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = 0.5 * k1_dev[base + n];
  }

  double k2[3];
  dexp_inv_so3(phi, v2, k2);

  for (auto n = 0; n < 3; ++n) {
    k2_dev[base + n] = k2[n];
  }

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = s_init_dev[base + n];
  }

  double phi2[3];
  for (auto n = 0; n < 3; ++n) {
    phi2[n] = 0.5 * k2[n];
  }

  double s_out[3];
  rodrigues_rotate(phi2, s_init, s_out);

  for (auto n = 0; n < 3; ++n) {
    s_out_dev[base + n] = s_out[n];
  }
}

__global__ void cuda_llg_rkmk4_kernel_step_3
(
  const double * s_init_dev,
  const double * s_step_dev,
  const double * k2_dev,
  double * k3_dev,
  double * s_out_dev,
  const jams::Real * h_step_dev,
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const unsigned dev_num_spins,
  const double dt
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned int base = 3u * idx;

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = h_step_dev[base + n] / mus_dev[idx];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = s_step_dev[base + n];
  }

  double omega[3];
  omega_llg(s, h, gyro_dev[idx], alpha_dev[idx], omega);

  double v3[3];
  for (auto n = 0; n < 3; ++n) {
    v3[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = 0.5 * k2_dev[base + n];
  }

  double k3[3];
  dexp_inv_so3(phi, v3, k3);

  for (auto n = 0; n < 3; ++n) {
    k3_dev[base + n] = k3[n];
  }

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = s_init_dev[base + n];
  }

  double s_out[3];
  rodrigues_rotate(k3, s_init, s_out);

  for (auto n = 0; n < 3; ++n) {
    s_out_dev[base + n] = s_out[n];
  }
}

__global__ void cuda_llg_rkmk4_kernel_step_4
(
  const double * s_init_dev,
  const double * s_step_dev,
  const double * k1_dev,
  const double * k2_dev,
  const double * k3_dev,
  double * s_out_dev,
  const jams::Real * h_step_dev,
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const unsigned dev_num_spins,
  const double dt
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned int base = 3u * idx;

  jams::Real h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = h_step_dev[base + n] / mus_dev[idx];
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = s_step_dev[base + n];
  }

  double omega[3];
  omega_llg(s, h, gyro_dev[idx], alpha_dev[idx], omega);

  double v4[3];
  for (auto n = 0; n < 3; ++n) {
    v4[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = k3_dev[base + n];
  }

  double k4[3];
  dexp_inv_so3(phi, v4, k4);

  double k[3];
  for (auto n = 0; n < 3; ++n) {
    k[n] = (k1_dev[base + n] + 2.0 * k2_dev[base + n] + 2.0 * k3_dev[base + n] + k4[n]) / 6.0;
  }

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = s_init_dev[base + n];
  }

  double s_out[3];
  rodrigues_rotate(k, s_init, s_out);

  for (auto n = 0; n < 3; ++n) {
    s_out_dev[base + n] = s_out[n];
  }
}

void CUDALLGRKMK4Solver::initialize(const libconfig::Setting& settings)
{
  // convert input in seconds to picoseconds for internal units
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  std::cout << "\ntimestep (ps) " << step_size_ << "\n";
  std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  std::string thermostat_name = jams::config_optional<std::string>(
    globals::config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  // Strang splitting means the thermostat timestep is 1/2 of the RKMK time step
  register_thermostat(Thermostat::create(thermostat_name, 0.5 * this->time_step()));

  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";
  std::cout << "done\n";

  s_init_.resize(globals::num_spins, 3);
  k1_.resize(globals::num_spins, 3);
  k2_.resize(globals::num_spins, 3);
  k3_.resize(globals::num_spins, 3);
}

void CUDALLGRKMK4Solver::run()
{
  const double t0 = time_;
  const double half_dt = 0.5 * step_size_;

  const dim3 block_size = {256, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  update_thermostat();
  thermostat_->record_done();
  thermostat_->wait_on(jams::instance().cuda_master_stream().get());

  cuda_llg_noise_step_rodrigues_kernel<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    globals::s.device_data(),
    thermostat_->device_data(),
    globals::gyro.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  cudaMemcpyAsync(s_init_.device_data(),
                globals::s.device_data(),
                globals::s.bytes(),
                cudaMemcpyDeviceToDevice,
                jams::instance().cuda_master_stream().get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  compute_fields();

  cuda_llg_rkmk4_kernel_step_1<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    k1_.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  time_ = t0 + half_dt;

  compute_fields();

  cuda_llg_rkmk4_kernel_step_2<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    globals::s.device_data(),
    k1_.device_data(),
    k2_.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();


  compute_fields();

  cuda_llg_rkmk4_kernel_step_3<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    globals::s.device_data(),
    k2_.device_data(),
    k3_.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  time_ = t0 + step_size_;

  compute_fields();

  cuda_llg_rkmk4_kernel_step_4<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    globals::s.device_data(),
    k1_.device_data(),
    k2_.device_data(),
    k3_.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  update_thermostat();
  thermostat_->record_done();
  thermostat_->wait_on(jams::instance().cuda_master_stream().get());


  cuda_llg_noise_step_rodrigues_kernel<<<grid_size, block_size, 0,  jams::instance().cuda_master_stream().get()>>>(
    globals::s.device_data(),
    thermostat_->device_data(),
    globals::gyro.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  iteration_++;
  time_ = iteration_ * step_size_;
}
