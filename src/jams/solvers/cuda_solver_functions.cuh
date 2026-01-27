//
// Created by Joseph Barker on 07/01/2026.
//

#ifndef JAMS_CUDA_SOLVER_FUNCTIONS_CUH
#define JAMS_CUDA_SOLVER_FUNCTIONS_CUH

#include "jams/cuda/cuda_device_vector_ops.h"
#include <cuda_runtime.h>

__device__ __forceinline__
void omega_llg(const double s[3], const jams::Real h[3],
               const jams::Real gyro, const jams::Real alpha,
               double result[3])
{

  result[0] = gyro * (h[0] + alpha * (s[1] * h[2] - s[2] * h[1]));
  result[1] = gyro * (h[1] + alpha * (s[2] * h[0] - s[0] * h[2]));
  result[2] = gyro * (h[2] + alpha * (s[0] * h[1] - s[1] * h[0]));
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
  // Numerically stable Rodrigues action using
  //   exp([phi])S = S + a (phi×S) + b (phi×(phi×S))
  // with a = sin(th)/th, b = (1-cos(th))/th^2.
  // Uses series for small th to avoid catastrophic cancellation.
  const double th2 = norm_squared(phi);

  // For double precision, cancellation in (1-cos(th))/th^2 becomes noticeable
  // well before th2 ~ 1e-24. Use a larger threshold and a series expansion.
  if (th2 < 1e-8)
  {
    // a = 1 - th^2/6 + th^4/120 + O(th^6)
    // b = 1/2 - th^2/24 + th^4/720 + O(th^6)
    const double th4 = th2 * th2;
    const double a = 1.0 - th2 * (1.0 / 6.0) + th4 * (1.0 / 120.0);
    const double b = 0.5 - th2 * (1.0 / 24.0) + th4 * (1.0 / 720.0);

    double c1[3];
    cross_product(phi, S, c1);
    double c2[3];
    cross_product(phi, c1, c2);

    for (int n = 0; n < 3; ++n)
    {
      result[n] = S[n] + a * c1[n] + b * c2[n];
    }
    return;
  }

  const double th = sqrt(th2);

  double s, c;
  sincos(th, &s, &c);

  const double a = s / th;
  const double b = (1.0 - c) / th2;

  double c1[3];
  cross_product(phi, S, c1);
  double c2[3];
  cross_product(phi, c1, c2);

  for (int n = 0; n < 3; ++n)
  {
    result[n] = S[n] + a * c1[n] + b * c2[n];
  }
}


__device__ inline void dexp_inv_so3(const double phi[3], const double v[3], double result[3])
{
  // dexp^{-1}_phi(v) for so(3) in vector form.
  // Uses: v - 1/2 phi×v + beta(th) phi×(phi×v)
  // beta(th) = (1/th^2) * (1 - (th/2) cot(th/2))
  // For small th, beta(th) = 1/12 + th^2/720 + th^4/30240 + O(th^6)

  const double th2 = norm_squared(phi);

  double c1[3];
  cross_product(phi, v, c1);
  double c2[3];
  cross_product(phi, c1, c2);

  // Use a larger threshold than 1e-24 to avoid loss of significance in
  // 1 - (th/2)cot(th/2) and to keep beta accurate.
  if (th2 < 1e-8)
  {
    const double th4 = th2 * th2;
    const double beta = (1.0 / 12.0) + th2 * (1.0 / 720.0) + th4 * (1.0 / 30240.0);

    for (int n = 0; n < 3; ++n)
    {
      result[n] = v[n] - 0.5 * c1[n] + beta * c2[n];
    }
    return;
  }

  const double th = sqrt(th2);
  const double half = 0.5 * th;

  double s, c;
  sincos(half, &s, &c);

  // beta = (1/th^2) * (1 - half * cot(half))
  // Use cot = cos/sin; safe here because we handled small angles above.
  const double cot_half = c / s;
  const double beta = (1.0 / th2) * (1.0 - half * cot_half);

  for (int n = 0; n < 3; ++n)
  {
    result[n] = v[n] - 0.5 * c1[n] + beta * c2[n];
  }
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
  rodrigues_rotate(phi, s, out);

  s_inout_dev[base+0] = out[0];
  s_inout_dev[base+1] = out[1];
  s_inout_dev[base+2] = out[2];
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