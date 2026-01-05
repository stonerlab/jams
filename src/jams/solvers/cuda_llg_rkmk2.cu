//
// Created by Joseph Barker on 05/01/2026.
//


#include "jams/solvers/cuda_llg_rkmk2.h"

#include "jams/core/globals.h"
#include "jams/cuda/cuda_device_vector_ops.h"


__device__ void rodrigues_rotate(const double phi[3], const double S[3], double result[3])
{
  const double th2 = norm_squared(phi);
  if (th2 < 1e-24)
  {
    // exp([phi])S ≈ S + phi×S + 1/2 phi×(phi×S)
    double c1[3];
    cross_product(phi, S, c1);
    double c2[3];
    cross_product(phi, c1, c2);

    for (auto n = 0; n < 3; ++n)
    {
      result[n] = S[n] + c1[n] + c2[n] * 0.5;
    }
    return;
  }

  const double th = sqrt(th2);
  const double s = sin(th);
  const double c = cos(th);

  double k[3];
  for (auto n = 0; n < 3; ++n)
  {
    k[n] = phi[n] * (1.0 / th);
  }

  // Rv = v c + (k×v) s + k (k·v)(1-c)
  double kxS[3];
  cross_product(k, S, kxS);
  const double kdotS = dot(k, S);
  for (auto n = 0; n < 3; ++n)
  {
    result[n] = S[n] * c + kxS[n] * s + kdotS * (1.0 - c) * k[n];
  }
  return;
}


// dexp^{-1}_phi(v) for so(3) in vector form.
// Uses: v - 1/2 phi×v + beta(th) phi×(phi×v)
// beta(th) = (1/th^2) * (1 - (th/2) cot(th/2))
__device__ void dexp_inv_so3(const double phi[3], const double v[3], double result[3]) {
  const double th2 = norm_squared(phi);
  if (th2 < 1e-24) {
    // dexp^{-1}_phi(v) = v - 1/2 (phi×v) + 1/12 (phi×(phi×v)) + O(||phi||^3)
    double c1[3];
    cross_product(phi, v, c1);
    double c2[3];
    cross_product(phi, c1, c2);

    for (auto n = 0; n < 3; ++n) {
      result[n] = v[n] - c1[n] * 0.5 + c2[n] * (1.0/12.0);
    }
    return;
  }

  const double th = sqrt(th2);
  const double half = 0.5 * th;
  const double cot_half = cos(half) / sin(half);
  const double beta = (1.0/th2) * (1.0 - half * cot_half);

  double c1[3];
  cross_product(phi, v, c1);
  double c2[3];
  cross_product(phi, c1, c2);

  for (auto n = 0; n < 3; ++n) {
    result[n] = v[n] - c1[n] * 0.5 + c2[n] * beta;
  }
  return;
}

__global__ void cuda_llg_rkmk2_kernel_step_1
(
  const double * s_init_dev,
  double * phi_dev,
  double * s_out_dev,
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
    s[n] = s_init_dev[base + n];
  }

  double sxh[3];
  cross_product(s, h, sxh);

  double omega[3];
  for (auto n = 0; n < 3; ++n) {
    omega[n] = gyro_dev[idx] * (h[n] + alpha_dev[idx] * sxh[n]);
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n) {
    phi[n] = omega[n] * dt * 0.5;
  }

  for (auto n = 0; n < 3; ++n) {
    phi_dev[base + n] = phi[n];
  }

  double s_out[3];
  rodrigues_rotate(phi, s, s_out);
  
  for (auto n = 0; n < 3; ++n) {
    s_out_dev[base + n] = s_out[n];
  }
}


__global__ void cuda_llg_rkmk2_kernel_step_2
(
  const double * s_init_dev,
  const double * s_step_dev,
  const double * phi_dev,
  double * s_out_dev,
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

  double omega[3];
  for (auto n = 0; n < 3; ++n) {
    omega[n] = gyro_dev[idx] * (h[n] + alpha_dev[idx] * sxh[n]);
  }

  double v2[3];
  for (auto n = 0; n < 3; ++n) {
    v2[n] = omega[n] * dt;
  }

  double phi[3];
  for (auto n = 0; n < 3; ++n)
  {
    phi[n] = phi_dev[base + n];
  }

  double k[3];
  dexp_inv_so3(phi, v2, k);

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


void CUDALLGRKMK2Solver::initialize(const libconfig::Setting& settings)
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

  phi_.resize(globals::num_spins, 3);
  s_init_.resize(globals::num_spins, 3);
  for (auto i = 0; i < globals::num_spins; ++i) {
    for (auto j = 0; j < 3; ++j) {
      s_init_(i, j) = globals::s(i, j);
    }
  }
}


void CUDALLGRKMK2Solver::run()
{
  double t0 = time_;

  const dim3 block_size = {64, 1, 1};
  auto grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(globals::num_spins), 1, 1});

  cudaMemcpyAsync(s_init_.device_data(),           // void *               dst
                  globals::s.device_data(),               // const void *         src
                  globals::s.bytes(),   // size_t               count
                  cudaMemcpyDeviceToDevice,    // enum cudaMemcpyKind  kind
                  dev_stream_.get());                   // device stream

  DEBUG_CHECK_CUDA_ASYNC_STATUS

  update_thermostat();
  compute_fields();

  cuda_llg_rkmk2_kernel_step_1<<<grid_size, block_size>>>(
    s_init_.device_data(),
    phi_.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    thermostat_->device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_
    );
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  double mid_time_step = 0.5 * step_size_;
  time_ = t0 + mid_time_step;

  compute_fields();

  cuda_llg_rkmk2_kernel_step_2<<<grid_size, block_size>>>(
    s_init_.device_data(),
    globals::s.device_data(),
    phi_.device_data(),
    globals::s.device_data(),
    globals::h.device_data(),
    thermostat_->device_data(),
    globals::gyro.device_data(),
    globals::mus.device_data(),
    globals::alpha.device_data(),
    globals::num_spins, step_size_
    );
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  iteration_++;
  time_ = iteration_ * step_size_;

  cudaDeviceSynchronize();

}