//
// Created by Joseph Barker on 05/01/2026.
//

#include "jams/solvers/cuda_llg_semi_implicit.h"

#include "jams/common.h"
#include "jams/core/globals.h"
#include "jams/cuda/cuda_device_vector_ops.h"

__device__ __forceinline__
void project_to_tangent(
  const double A[3],
  const double S[3],
  double result[3])
{
  const double S_dot_A = dot(S, A);
  #pragma unroll
  for (auto n = 0; n < 3; ++n) {
    result[n] = A[n] - S_dot_A * S[n];
  }
}


__device__ __forceinline__
void cayley(const double A[3],
            const double S[3],
            double result[3])
{
  // AxS
  double AxS[3];
  cross_product(A, S,AxS);

  // A x (A x S)
  double AxAxS[3];
  cross_product(A, AxS,AxAxS);

  const double scale = 1.0 / (1.0 + 0.25 * norm_squared(A));

  #pragma unroll
  for (auto n = 0; n < 3; ++n) {
    result[n] = S[n] + (AxS[n]+ 0.5 * AxAxS[n]) * scale;
  }
}


__global__ void cuda_llg_semi_implicit_kernel_mid_step(
  const unsigned num_spins,
  const double* __restrict__ s_init_dev,
  const double* __restrict__ s_pred_dev,
  double* __restrict__ s_mid_dev
)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) return;

  const unsigned base = 3u * idx;

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = s_init_dev[base + n] + s_pred_dev[base + n];
  }

  const double inv_norm = rsqrt(norm_squared(s));

  for (auto n = 0; n < 3; ++n) {
    s_mid_dev[base + n] = s[n] * inv_norm;
  }
}


__global__ void cuda_llg_semi_implicit_kernel_step
(
  const double * s_init_dev, // S_n
  const double * s_step_dev, // S at this step (S_n for predictor, (S_n + S_n+1) / 2 for corrector)
  double * s_out_dev,        // S_n+1
  const double * h_step_dev,  // field at the same time as s_step
  const jams::Real * noise_init_dev, // noise at the same time as s_init
  const double * gyro_dev,
  const double * mus_dev,
  const double * alpha_dev,
  const unsigned dev_num_spins,
  const double dt
)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dev_num_spins) return;

  const unsigned int base = 3u * idx;

  double h[3];
  for (auto n = 0; n < 3; ++n) {
    h[n] = ((h_step_dev[base + n] / mus_dev[idx]) + noise_init_dev[base + n]);
  }

  double s[3];
  for (auto n = 0; n < 3; ++n) {
    s[n] = s_step_dev[base + n];
  }

  double sxh[3];
  cross_product(s, h, sxh);

  double A[3];
  for (auto n = 0; n < 3; ++n) {
    A[n] = dt * gyro_dev[idx] * (h[n] + alpha_dev[idx] * sxh[n]);
  }

  double A_perp[3];
  project_to_tangent(A, s, A_perp);

  double s_init[3];
  for (auto n = 0; n < 3; ++n) {
    s_init[n] = s_init_dev[base + n];
  }

  double s_out[3];
  cayley(A_perp, s_init, s_out);

  for (auto n = 0; n < 3; ++n) {
    s_out_dev[base + n] = s_out[n];
  }
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
  register_thermostat(Thermostat::create(thermostat_name, this->time_step()));

  std::cout << "  thermostat " << thermostat_name.c_str() << "\n";

  std::cout << "done\n";

  s_pred_.resize(globals::num_spins, 3);
  s_init_.resize(globals::num_spins, 3);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_init_(i, j) = globals::s(i, j);
    }
  }
}


void CUDALLGSemiImplictSolver::run()
{
  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cudaMemcpyAsync(s_init_.device_data(),           // void *               dst
                  globals::s.device_data(),               // const void *         src
                  globals::s.bytes(),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  jams::instance().cuda_master_stream().get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();
  compute_fields();

  cuda_llg_semi_implicit_kernel_step<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    globals::s.device_data(),
    s_pred_.device_data(),
    globals::h.device_data(),
    thermostat_->device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_
    );
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cuda_llg_semi_implicit_kernel_mid_step<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    globals::num_spins,
    s_init_.device_data(),
    s_pred_.device_data(),
    globals::s.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  jams::instance().cuda_master_stream().synchronize();
  compute_fields();

  cuda_llg_semi_implicit_kernel_step<<<grid_size, block_size, 0, jams::instance().cuda_master_stream().get()>>>(
    s_init_.device_data(),
    globals::s.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    thermostat_->device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_
    );
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  jams::instance().cuda_master_stream().synchronize();

  iteration_++;
  time_ = iteration_ * step_size_;


}