//
// Created by Joseph Barker on 07/01/2026.
//

#ifndef JAMS_CUDA_SOLVER_FUNCTIONS_CUH
#define JAMS_CUDA_SOLVER_FUNCTIONS_CUH

#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/solvers/llg_rkmk_functions.h"
#include <cuda_runtime.h>

__device__ __forceinline__
void omega_llg(const double s[3], const jams::Real h[3],
               const jams::Real gyro, const jams::Real alpha,
               double result[3])
{
  jams::solvers::rkmk::omega_llg(s, h, gyro, alpha, result);
}


__device__ __forceinline__
double3 omega_llg(const double3& s, const jams::Real3& h,
               const double gyro, const double alpha)
{
  return {
    gyro * (alpha * (s.y * h.z - s.z * h.y) + h.x),
    gyro * (alpha * (s.z * h.x - s.x * h.z) + h.y),
    gyro * (alpha * (s.x * h.y - s.y * h.x) + h.z),
  };
}


__device__ __forceinline__
double3 project_to_tangent(const double3& A, const double3& S)
{
  const double S_dot_A = S.x*A.x + S.y*A.y + S.z*A.z;
  return {
    __fma_rn(-S_dot_A, S.x, A.x),
    __fma_rn(-S_dot_A, S.y, A.y),
    __fma_rn(-S_dot_A, S.z, A.z)
  };
}

__device__ __forceinline__
double3 cayley_rotate(const double3& A, const double3& S)
{
  // AxS
  double3 AxS = cross_product(A, S);

  const double norm_sq = A.x*A.x + A.y*A.y + A.z*A.z;
  const double scale = 1.0 / (1.0 + 0.25 * norm_sq);

  return {
    __fma_rn((AxS.x + 0.5 * (A.y * AxS.z - A.z * AxS.y)), scale, S.x),
    __fma_rn((AxS.y + 0.5 * (A.z * AxS.x - A.x * AxS.z)), scale, S.y),
    __fma_rn((AxS.z + 0.5 * (A.x * AxS.y - A.y * AxS.x)), scale, S.z)
  };
}

__device__ __forceinline__
void cayley_rotate(const double A[3], const double S[3], double result[3])
{
  // AxS
  double AxS[3];
  cross_product(A, S, AxS);

  const double norm_sq = norm_squared(A);
  // const double scale = 1.0 / __fma_rn(0.25, norm_sq, 1.0);
  const double scale = 1.0 / (1.0 + 0.25 * norm_sq);

  result[0] = __fma_rn((AxS[0] + 0.5 * (A[1] * AxS[2] - A[2] * AxS[1])), scale, S[0]);
  result[1] = __fma_rn((AxS[1] + 0.5 * (A[2] * AxS[0] - A[0] * AxS[2])), scale, S[1]);
  result[2] = __fma_rn((AxS[2] + 0.5 * (A[0] * AxS[1] - A[1] * AxS[0])), scale, S[2]);
}


__device__ inline void rodrigues_rotate(const double phi[3], const double S[3], double result[3])
{
  jams::solvers::rkmk::rodrigues_rotate(phi, S, result);
}


__device__ inline void dexp_inv_so3(const double phi[3], const double v[3], double result[3])
{
  jams::solvers::rkmk::dexp_inv_so3(phi, v, result);
}

__device__ __forceinline__ void rkmk_store_spin_and_cache
(
  double *s_out_dev,
  jams::Real *s_cache_dev,
  const unsigned base,
  const double s[3]
)
{
  for (auto n = 0; n < 3; ++n) {
    s_out_dev[base + n] = s[n];
  }

  if (s_cache_dev != nullptr) {
    for (auto n = 0; n < 3; ++n) {
      s_cache_dev[base + n] = static_cast<jams::Real>(s[n]);
    }
  }
}

__device__ __forceinline__ void rkmk_noise_step_rodrigues
(
  const double s[3],
  const jams::Real *noise_dev,
  const jams::Real *gyro_dev,
  const jams::Real *alpha_dev,
  const unsigned idx,
  const unsigned base,
  const double dt,
  double out[3]
)
{
  const jams::Real h[3] = {
    noise_dev[base + 0],
    noise_dev[base + 1],
    noise_dev[base + 2]
  };

  double w[3];
  omega_llg(s, h, gyro_dev[idx], alpha_dev[idx], w);

  const double phi[3] = {dt * w[0], dt * w[1], dt * w[2]};
  rodrigues_rotate(phi, s, out);
}

__global__ inline void cuda_llg_noise_step_rodrigues_cache_kernel(
  double* s_inout_dev,
  jams::Real* s_cache_dev,
  const jams::Real* noise_dev,
  const jams::Real* gyro_dev,
  const jams::Real* alpha_dev,
  unsigned num_spins,
  double dt)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) return;
  const unsigned base = 3u * idx;

  const double s[3] = {
    s_inout_dev[base + 0],
    s_inout_dev[base + 1],
    s_inout_dev[base + 2]
  };

  double out[3];
  rkmk_noise_step_rodrigues(s, noise_dev, gyro_dev, alpha_dev, idx, base, dt, out);
  rkmk_store_spin_and_cache(s_inout_dev, s_cache_dev, base, out);
}


__global__ inline void cuda_llg_noise_step_rodrigues_kernel(
  double* s_inout_dev,
  const jams::Real* noise_dev,
  const jams::Real* gyro_dev,
  const jams::Real* alpha_dev,
  unsigned num_spins,
  double dt)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) return;
  const unsigned base = 3u * idx;

  const double s[3] = {
    s_inout_dev[base + 0],
    s_inout_dev[base + 1],
    s_inout_dev[base + 2]
  };

  double out[3];
  rkmk_noise_step_rodrigues(s, noise_dev, gyro_dev, alpha_dev, idx, base, dt, out);
  rkmk_store_spin_and_cache(s_inout_dev, nullptr, base, out);
}

__global__ inline void cuda_llg_noise_step_cayley_kernel(
  double* s_inout_dev,
  const jams::Real* noise_dev,
  const jams::Real* gyro_dev,
  const jams::Real* alpha_dev,
  unsigned num_spins,
  double dt)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) return;
  const unsigned base = 3u * idx;

  // Spin
  double s[3] = {s_inout_dev[base+0], s_inout_dev[base+1], s_inout_dev[base+2]};

  // Treat white noise as an effective field for this substep
  jams::Real h[3] = {
    noise_dev[base+0],
    noise_dev[base+1],
    noise_dev[base+2]
  };

  double w[3];
  omega_llg(s, h, gyro_dev[idx], alpha_dev[idx], w);

  double phi[3] = {dt * w[0], dt * w[1], dt * w[2]};
  double out[3];

  cayley_rotate(phi, s, out);

  s_inout_dev[base+0] = out[0];
  s_inout_dev[base+1] = out[1];
  s_inout_dev[base+2] = out[2];
}



// __global__ inline void cuda_llg_noise_step_cayley_kernel(
//   double* __restrict__ s_inout_dev,
//   const jams::Real* __restrict__ noise_dev,
//   const jams::Real* __restrict__ gyro_dev,
//   const jams::Real* __restrict__ alpha_dev,
//   unsigned num_spins,
//   jams::Real dt)
// {
//   const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= num_spins) return;
//   const unsigned base = 3u * idx;
//
//   const double3 s = {
//     s_inout_dev[base + 0],
//     s_inout_dev[base + 1],
//     s_inout_dev[base + 2]
//   };
//
//   const jams::Real gyro = gyro_dev[idx];
//   const jams::Real3 h = {
//     noise_dev[base + 0] * dt * gyro,
//     noise_dev[base + 1] * dt * gyro,
//     noise_dev[base + 2] * dt * gyro
//   };
//
//   const jams::Real alpha = alpha_dev[idx];
//   double3 omega = {
//       alpha * (s.y * h.z - s.z * h.y) + h.x,
//       alpha * (s.z * h.x - s.x * h.z) + h.y,
//       alpha * (s.x * h.y - s.y * h.x) + h.z,
//   };
//
//   // This projection is not strictly necessary but can help to reduce
//   // errors in the spin norm due to floating point arithmetic.
//   omega = project_to_tangent(omega, s);
//
//   double3 out = cayley_rotate(omega, s);
//
//   s_inout_dev[base+0] = out.x;
//   s_inout_dev[base+1] = out.y;
//   s_inout_dev[base+2] = out.z;
// }

#endif //JAMS_CUDA_SOLVER_FUNCTIONS_CUH
