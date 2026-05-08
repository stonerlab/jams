#ifndef JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH
#define JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH

#include <jams/hamiltonian/tesseral_polynomial_evaluator.h>

__global__ void cuda_crystal_field_energy_kernel(
    const unsigned int num_spins,
    const jams::RealHi* dev_s,
    const jams::Real* dev_u_axes,
    const jams::Real* dev_v_axes,
    const jams::Real* dev_w_axes,
    const int* dev_spin_pointer,
    const int* dev_tesseral_keys,
    const jams::Real* dev_tesseral_coefficients,
    const jams::Real* dev_axial_coefficients,
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

  dev_e[idx] = jams::tesseral_polynomial::energy_for_spin_with_axial_terms(
      idx,
      sx,
      sy,
      sz,
      dev_u_axes,
      dev_v_axes,
      dev_w_axes,
      dev_spin_pointer,
      dev_tesseral_keys,
      dev_tesseral_coefficients,
      dev_axial_coefficients);
}

__global__ void cuda_crystal_field_kernel(
    const unsigned int num_spins,
    const jams::RealHi* dev_s,
    const jams::Real* dev_u_axes,
    const jams::Real* dev_v_axes,
    const jams::Real* dev_w_axes,
    const int* dev_spin_pointer,
    const int* dev_tesseral_keys,
    const jams::Real* dev_tesseral_coefficients,
    const jams::Real* dev_axial_coefficients,
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
  jams::tesseral_polynomial::field_for_spin_with_axial_terms(
      idx,
      sx,
      sy,
      sz,
      dev_u_axes,
      dev_v_axes,
      dev_w_axes,
      dev_spin_pointer,
      dev_tesseral_keys,
      dev_tesseral_coefficients,
      dev_axial_coefficients,
      h);

  dev_h[base + 0] = static_cast<jams::Real>(h[0]);
  dev_h[base + 1] = static_cast<jams::Real>(h[1]);
  dev_h[base + 2] = static_cast<jams::Real>(h[2]);
}

#endif // JAMS_HAMILTONIAN_CUDA_CRYSTAL_FIELD_KERNEL_CUH
