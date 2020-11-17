#ifndef JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_KERNEL_CUH
#define JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_KERNEL_CUH

#include <array>

__global__ void cuda_applied_field_energy_kernel(const unsigned int num_spins, const double * dev_s, const double * dev_mus, const double3 b_field, double * dev_e) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {
    dev_e[idx] = -dev_mus[idx] * (dev_s[3 * idx + 0] * b_field.x + dev_s[3 * idx + 1] * b_field.y + dev_s[3 * idx + 2] * b_field.z);
  }
}

__global__ void cuda_applied_field_kernel(const unsigned int num_spins, const double * dev_mus, const double3 b_field, double * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {
    dev_h[3*idx + 0] = dev_mus[idx] * b_field.x;
    dev_h[3*idx + 1] = dev_mus[idx] * b_field.y;
    dev_h[3*idx + 2] = dev_mus[idx] * b_field.z;
  }
}

#endif //JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_KERNEL_CUH
