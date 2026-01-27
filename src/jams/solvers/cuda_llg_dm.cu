// cuda_llg_dm.cu
//
// Depondt–Mertens (DM) scheme on GPU, in the same style as CUDALLGRKMK2Solver.
// Deterministic DM step (no noise) is:
//   1) ω_n = ω(S_n, H(S_n,t_n))
//   2) S*  = R(Δt ω_n) S_n
//   3) ω_* = ω(S*,  H(S*,t_{n+1}))   (or t_n+Δt if your fields depend explicitly on time)
//   4) S_{n+1} = R(Δt * 0.5(ω_n + ω_*)) S_n
//
// With Strang splitting for white noise (as you already do):
//   noise(Δt/2) -> DM_det(Δt) -> noise(Δt/2)

#include "jams/solvers/cuda_llg_dm.h"

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/solvers/cuda_solver_functions.cuh"

// -----------------------------------------------------------------------------
// Kernel 1: compute ω_n, store it, and do predictor rotation S* = R(Δt ω_n) S_n
// -----------------------------------------------------------------------------
__global__ void cuda_llg_dm_kernel_predict
(
  const double * s_init_dev,
  double * omega1_dev,                 // store ω_n (double, 3 per spin)
  double * s_pred_dev,                 // output S*
  const jams::Real * h_step_dev,        // field at time t_n for S_n
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const unsigned dev_num_spins,
  const double dt
)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned base = 3u * idx;

  jams::Real h[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    h[n] = h_step_dev[base + n] / mus_dev[idx];
  }

  double s[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    s[n] = s_init_dev[base + n];
  }

  double omega1[3];
  omega_llg(s, h, gyro_dev[idx], alpha_dev[idx], omega1);

  // store ω_n
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    omega1_dev[base + n] = omega1[n];
  }

  // predictor: S* = R(Δt ω_n) S_n
  double phi[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    phi[n] = omega1[n] * dt;
  }

  double s_pred[3];
  rodrigues_rotate(phi, s, s_pred);

  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    s_pred_dev[base + n] = s_pred[n];
  }
}


// -----------------------------------------------------------------------------
// Kernel 2: compute ω_* from S* and H(S*), average, then rotate original S_n
// -----------------------------------------------------------------------------
__global__ void cuda_llg_dm_kernel_correct
(
  const double * s_init_dev,            // original S_n
  const double * s_pred_dev,            // S*
  const double * omega1_dev,            // ω_n
  double * s_out_dev,                   // S_{n+1}
  const jams::Real * h_step_dev,        // field at time t_{n+1} (or t_n+Δt) for S*
  const jams::Real * gyro_dev,
  const jams::Real * mus_dev,
  const jams::Real * alpha_dev,
  const unsigned dev_num_spins,
  const double dt
)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned base = 3u * idx;

  jams::Real h[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    h[n] = h_step_dev[base + n] / mus_dev[idx];
  }

  double s_pred[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    s_pred[n] = s_pred_dev[base + n];
  }

  double omega2[3];
  omega_llg(s_pred, h, gyro_dev[idx], alpha_dev[idx], omega2);

  double omega1[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    omega1[n] = omega1_dev[base + n];
  }

  // average angular velocities (DM)
  double omega_bar[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    omega_bar[n] = 0.5 * (omega1[n] + omega2[n]);
  }

  // rotate original spin S_n with averaged ω
  double s_init[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    s_init[n] = s_init_dev[base + n];
  }

  double phi[3];
  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    phi[n] = omega_bar[n] * dt;
  }

  double s_out[3];
  rodrigues_rotate(phi, s_init, s_out);

  #pragma unroll
  for (int n = 0; n < 3; ++n) {
    s_out_dev[base + n] = s_out[n];
  }
}


// -----------------------------------------------------------------------------
// Solver class
// -----------------------------------------------------------------------------
void CUDALLGDMSolver::initialize(const libconfig::Setting& settings)
{
  step_size_ = jams::config_required<double>(settings, "t_step") / 1e-12;
  auto t_max = jams::config_required<double>(settings, "t_max") / 1e-12;
  auto t_min = jams::config_optional<double>(settings, "t_min", 0.0) / 1e-12;

  max_steps_ = static_cast<int>(t_max / step_size_);
  min_steps_ = static_cast<int>(t_min / step_size_);

  std::cout << "\ntimestep (ps) " << step_size_ << "\n";
  std::cout << "\nt_max (ps) " << t_max << " steps " << max_steps_ << "\n";
  std::cout << "\nt_min (ps) " << t_min << " steps " << min_steps_ << "\n";

  std::string thermostat_name =
    jams::config_optional<std::string>(globals::config->lookup("solver"),
                                       "thermostat",
                                       jams::defaults::solver_gpu_thermostat);

  register_thermostat(Thermostat::create(thermostat_name, 0.5 * this->time_step()));
  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  s_init_.resize(globals::num_spins, 3);
  s_pred_.resize(globals::num_spins, 3);
  omega1_.resize(globals::num_spins, 3);

  // initial snapshot (same as your RKMK2)
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_init_(i, j) = globals::s(i, j);
    }
  }
}


void CUDALLGDMSolver::run()
{
  const double t0 = time_;
  const double half_dt = 0.5 * step_size_;

  const dim3 block_size = {256, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  // --- Strang: noise half-step
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

  // snapshot S_n
  cudaMemcpyAsync(
    s_init_.device_data(),
    globals::s.device_data(),
    globals::s.bytes(),
    cudaMemcpyDeviceToDevice,
    jams::instance().cuda_master_stream().get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  // --- fields at t_n, S_n
  time_ = t0;
  compute_fields();

  // predictor: S* and store ω_n
  cuda_llg_dm_kernel_predict<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    omega1_.device_data(),
    s_pred_.device_data(),
    globals::h.device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  // --- fields at t_{n+1} using S* (this matches DM for time-dependent fields)
  // Put predicted spins into globals::s so compute_fields() uses them.
  cudaMemcpyAsync(
    globals::s.device_data(),
    s_pred_.device_data(),
    globals::s.bytes(),
    cudaMemcpyDeviceToDevice,
    jams::instance().cuda_master_stream().get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  time_ = t0 + step_size_;
  compute_fields();

  // correct: rotate original S_n using averaged ω, output into globals::s
  cuda_llg_dm_kernel_correct<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    s_pred_.device_data(),
    omega1_.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  // --- Strang: noise half-step
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

  iteration_++;
  time_ = iteration_ * step_size_;
}