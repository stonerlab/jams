#ifndef JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_KERNEL_CUH
#define JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_KERNEL_CUH

#include <array>

__global__ void cuda_applied_field_energy_kernel(const unsigned int num_spins, const double * dev_s, const double * dev_mus, const double b_field[3], double * dev_e) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_spins) {
    dev_e[idx] = -dev_mus[idx] * (dev_s[3 * idx + 0] * b_field[0] + dev_s[3 * idx + 1] * b_field[1] + dev_s[3 * idx + 2] * b_field[2]);
  }
}

__global__ void cuda_applied_field_kernel(const unsigned int num_spins, const double * dev_mus, const double b_field[3], double * dev_h) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int gxy = 3 * idx + idy;

  if (idx < num_spins && idy < 3) {
    dev_h[gxy] += dev_mus[idx] * b_field[idy];
  }
}

#endif //JAMS_HAMILTONIAN_CUDA_APPLIED_FIELD_KERNEL_CUH
