#ifndef JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH
#define JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH

#include <jams/maths/tesseral_harmonics.h>

namespace {

constexpr unsigned int kCrystalFieldNumTerms = 27;

__device__ inline void crystal_field_lm_from_index(const unsigned int index, int& l, int& m)
{
  if (index < 5) {
    l = 2;
    m = static_cast<int>(index) - 2;
  } else if (index < 14) {
    l = 4;
    m = static_cast<int>(index) - 9;
  } else {
    l = 6;
    m = static_cast<int>(index) - 20;
  }
}

} // namespace

__global__ void cuda_crystal_field_energy_kernel(
    const unsigned int num_spins,
    const double* dev_s,
    const double* dev_cf_coeffs,
    jams::Real* dev_e)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) {
    return;
  }

  const unsigned int base = 3u * idx;
  const double sx = dev_s[base + 0];
  const double sy = dev_s[base + 1];
  const double sz = dev_s[base + 2];

  double energy = 0.0;

#pragma unroll
  for (auto term = 0u; term < kCrystalFieldNumTerms; ++term) {
    const auto coefficient = dev_cf_coeffs[term * num_spins + idx];
    if (coefficient == 0.0) {
      continue;
    }

    int l;
    int m;
    crystal_field_lm_from_index(term, l, m);
    const auto scale = jams::tesseral_racah_normalisation_scale_lookup<double>(l, m);
    const auto key = jams::tesseral_key(l, m);
    energy += coefficient * scale * jams::tesseral_monic_polynomial_key_lookup(key, sx, sy, sz);
  }

  dev_e[idx] = static_cast<jams::Real>(energy);
}

__global__ void cuda_crystal_field_kernel(
    const unsigned int num_spins,
    const double* dev_s,
    const double* dev_cf_coeffs,
    jams::Real* dev_h)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) {
    return;
  }

  const unsigned int base = 3u * idx;
  const double sx = dev_s[base + 0];
  const double sy = dev_s[base + 1];
  const double sz = dev_s[base + 2];

  double h[3] = {0.0, 0.0, 0.0};

#pragma unroll
  for (auto term = 0u; term < kCrystalFieldNumTerms; ++term) {
    const auto coefficient = dev_cf_coeffs[term * num_spins + idx];
    if (coefficient == 0.0) {
      continue;
    }

    int l;
    int m;
    crystal_field_lm_from_index(term, l, m);
    const auto scale = jams::tesseral_racah_normalisation_scale_lookup<double>(l, m);
    const auto key = jams::tesseral_key(l, m);

    double grad[3];
    jams::tesseral_monic_polynomial_grad_key_lookup(key, sx, sy, sz, grad);
    const auto grad_x = scale * grad[0];
    const auto grad_y = scale * grad[1];
    const auto grad_z = scale * grad[2];
    const auto radial_grad = sx * grad_x + sy * grad_y + sz * grad_z;

    h[0] += coefficient * (radial_grad * sx - grad_x);
    h[1] += coefficient * (radial_grad * sy - grad_y);
    h[2] += coefficient * (radial_grad * sz - grad_z);
  }

  dev_h[base + 0] = static_cast<jams::Real>(h[0]);
  dev_h[base + 1] = static_cast<jams::Real>(h[1]);
  dev_h[base + 2] = static_cast<jams::Real>(h[2]);
}

#endif // JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH
