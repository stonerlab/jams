#ifndef JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH
#define JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH

#include <jams/hamiltonian/tesseral_polynomial_evaluator.h>

__global__ void cuda_crystal_field_energy_kernel(
    const unsigned int num_spins,
    const jams::RealHi* dev_s,
    const int* dev_spin_pointer,
    const int* dev_tesseral_keys,
    const jams::Real* dev_tesseral_coefficients,
    jams::Real* dev_e)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) {
    return;
  }

  const unsigned int base = 3u * idx;
  const jams::Real sx = static_cast<jams::Real>(dev_s[base + 0]);
  const jams::Real sy = static_cast<jams::Real>(dev_s[base + 1]);
  const jams::Real sz = static_cast<jams::Real>(dev_s[base + 2]);

  dev_e[idx] = static_cast<jams::Real>(
      jams::tesseral_polynomial::energy_from_local_terms(
          dev_spin_pointer[idx],
          dev_spin_pointer[idx + 1],
          dev_tesseral_keys,
          dev_tesseral_coefficients,
          sx,
          sy,
          sz));
}

__global__ void cuda_crystal_field_kernel(
    const unsigned int num_spins,
    const jams::RealHi* dev_s,
    const int* dev_spin_pointer,
    const int* dev_tesseral_keys,
    const jams::Real* dev_tesseral_coefficients,
    jams::Real* dev_h)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_spins) {
    return;
  }

  const unsigned int base = 3u * idx;
  const jams::Real sx = static_cast<jams::Real>(dev_s[base + 0]);
  const jams::Real sy = static_cast<jams::Real>(dev_s[base + 1]);
  const jams::Real sz = static_cast<jams::Real>(dev_s[base + 2]);

  jams::Real h[3] = {0.0, 0.0, 0.0};
  jams::tesseral_polynomial::negative_gradient_from_local_terms(
      dev_spin_pointer[idx],
      dev_spin_pointer[idx + 1],
      dev_tesseral_keys,
      dev_tesseral_coefficients,
      sx,
      sy,
      sz,
      h);

  dev_h[base + 0] = static_cast<jams::Real>(h[0]);
  dev_h[base + 1] = static_cast<jams::Real>(h[1]);
  dev_h[base + 2] = static_cast<jams::Real>(h[2]);
}

#endif // JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH
