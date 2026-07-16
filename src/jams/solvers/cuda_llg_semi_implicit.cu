//
// Created by Joseph Barker on 05/01/2026.
//

#include "jams/solvers/cuda_llg_semi_implicit.h"

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/solvers/cuda_solver_functions.cuh"


template <typename DtGyroMuParam, typename AlphaParam>
__global__ __maxnreg__(32)
void cuda_llg_semi_implicit_pred_kernel
(
  double * __restrict__ s_inout_dev, // in: S_n, out: (S_n + S'_n+1) / 2
  const jams::Real * __restrict__ h_dev,
  const DtGyroMuParam dt_gyro_mu,
  const AlphaParam alpha,
  const unsigned dev_num_spins
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned int base = 3u * idx;

  jams::Real3 h = {
    h_dev[base + 0],
    h_dev[base + 1],
    h_dev[base + 2]
  };

  const double3 s = {
    s_inout_dev[base + 0],
    s_inout_dev[base + 1],
    s_inout_dev[base + 2]
  };

  double3 omega = omega_llg(s, h, dt_gyro_mu.get(idx), alpha.get(idx));

  omega = project_to_tangent(omega, s);

  double3 s_pred = cayley_rotate(omega, s);

  // Make the midpoint estimate
  s_pred.x = 0.5 * (s_pred.x + s.x);
  s_pred.y = 0.5 * (s_pred.y + s.y);
  s_pred.z = 0.5 * (s_pred.z + s.z);

  // Write (S_{n} + S'_{n+1}) / 2 to memory
  s_inout_dev[base + 0] = s_pred.x;
  s_inout_dev[base + 1] = s_pred.y;
  s_inout_dev[base + 2] = s_pred.z;
}

template <typename DtGyroMuParam, typename AlphaParam>
__global__ void cuda_llg_semi_implicit_corr_kernel
(
  double * __restrict__ s_inout_dev, // in: (S_n + S'_{n+1}) / 2 out: S_{n+1}
  const double * __restrict__ s_init_dev, // S_n
  const jams::Real * __restrict__ h_dev,  // field at the same time as s_step
  const DtGyroMuParam dt_gyro_mu,
  const AlphaParam alpha,
  const unsigned dev_num_spins
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned int base = 3u * idx;

  const jams::Real3 h = {
    h_dev[base + 0],
    h_dev[base + 1],
    h_dev[base + 2]
  };

  const double3 s = {
    s_inout_dev[base + 0],
    s_inout_dev[base + 1],
    s_inout_dev[base + 2]
  };

  double3 omega = omega_llg(s, h, dt_gyro_mu.get(idx), alpha.get(idx));

  omega = project_to_tangent(omega, s);

  double3 s_out = {
    s_init_dev[base + 0],
    s_init_dev[base + 1],
    s_init_dev[base + 2]
  };

  s_out = cayley_rotate(omega, s_out);

  s_inout_dev[base + 0] = s_out.x;
  s_inout_dev[base + 1] = s_out.y;
  s_inout_dev[base + 2] = s_out.z;
}


void CUDALLGSemiImplictSolver::initialize(const libconfig::Setting& settings)
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

  std::string thermostat_name = jams::config_optional<std::string>(globals::config->lookup("solver"), "thermostat", jams::defaults::solver_gpu_thermostat);
  register_thermostat(Thermostat::create(thermostat_name, 0.5 * this->time_step()));

  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  std::cout << "done\n";

  initialize_gyro_eff(settings, gyro_eff_);
  const auto gyro_eff_choice = cuda_spin_parameter_choice(gyro_eff_);
  gyro_eff_is_uniform_ = gyro_eff_choice.is_uniform;
  gyro_eff_uniform_value_ = gyro_eff_choice.uniform_value;
  const auto alpha_choice = cuda_spin_parameter_choice(globals::alpha);
  alpha_is_uniform_ = alpha_choice.is_uniform;
  alpha_uniform_value_ = alpha_choice.uniform_value;

  dt_gyro_mu_.resize(globals::num_spins);
  for (auto i = 0; i < globals::num_spins; ++i)
  {
    dt_gyro_mu_(i) = step_size_ * gyro_eff_(i) * globals::inv_mus(i);
  }
  const auto dt_gyro_mu_choice = cuda_spin_parameter_choice(dt_gyro_mu_);
  dt_gyro_mu_is_uniform_ = dt_gyro_mu_choice.is_uniform;
  dt_gyro_mu_uniform_value_ = dt_gyro_mu_choice.uniform_value;

  s_init_.resize(globals::num_spins, 3);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_init_(i, j) = globals::s(i, j);
    }
  }
}


template <typename GyroParam, typename AlphaParam, typename DtGyroMuParam>
void CUDALLGSemiImplictSolver::run_with_parameters(
    const GyroParam gyro,
    const AlphaParam alpha,
    const DtGyroMuParam dt_gyro_mu)
{
  jams::Real t0 = time_;
  const jams::Real half_dt = 0.5 * step_size_;

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  update_thermostat();
  thermostat_->record_done();
  thermostat_->wait_on(jams::instance().cuda_master_stream().get());

  cuda_llg_noise_step_cayley_kernel<<<grid_size, block_size, 0,  jams::instance().cuda_master_stream().get()>>>(
    globals::s.mutable_device_data(),
    thermostat_->device_data(),
    gyro,
    alpha,
    globals::num_spins, half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(jams::instance().cuda_master_stream().get());
  record_spin_barrier_event();

  cudaMemcpyAsync(s_init_.mutable_device_data(),           // void *               dst
                globals::s.device_data(),               // const void *         src
                globals::s.bytes(),   // size_t               count
                cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                jams::instance().cuda_master_stream().get());                   // device stream
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  compute_fields();

  cuda_llg_semi_implicit_pred_kernel<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    globals::s.mutable_device_data(),
    globals::h.device_data(),
    dt_gyro_mu,
    alpha,
    globals::num_spins
    );
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  record_spin_barrier_event();

  jams::Real mid_time_step = 0.5 * step_size_;
  time_ = t0 + mid_time_step;
  compute_fields();

  cuda_llg_semi_implicit_corr_kernel<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    globals::s.mutable_device_data(),
    s_init_.device_data(),
    globals::h.device_data(),
    dt_gyro_mu,
    alpha,
    globals::num_spins
    );
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();
  thermostat_->record_done();
  thermostat_->wait_on(jams::instance().cuda_master_stream().get());

  cuda_llg_noise_step_cayley_kernel<<<grid_size, block_size, 0,  jams::instance().cuda_master_stream().get()>>>(
    globals::s.mutable_device_data(),
    thermostat_->device_data(),
    gyro,
    alpha,
    globals::num_spins, half_dt);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  thermostat_->record_consumed(jams::instance().cuda_master_stream().get());
  record_spin_barrier_event();

  iteration_++;
  time_ = iteration_ * step_size_;
}

void CUDALLGSemiImplictSolver::run()
{
  dispatch_cuda_spin_parameter(
      {gyro_eff_is_uniform_, gyro_eff_uniform_value_},
      gyro_eff_,
      [this](const auto gyro) {
        dispatch_cuda_spin_parameters(
            {dt_gyro_mu_is_uniform_, dt_gyro_mu_uniform_value_},
            {alpha_is_uniform_, alpha_uniform_value_},
            dt_gyro_mu_,
            globals::alpha,
            [this, gyro](const auto dt_gyro_mu, const auto alpha) {
              run_with_parameters(gyro, alpha, dt_gyro_mu);
            });
      });
}
