//
// Created by Joe Barker on 2018/05/29.
//
#ifndef JAMS_RANDOM_ANISOTROPY_CUDA_KERNEL_H
#define JAMS_RANDOM_ANISOTROPY_CUDA_KERNEL_H

#include "jams/cuda/cuda_device_vector_ops.h"

__global__
void random_anisotropy_cuda_field_kernel(
        const int num_spins,
        const double * spins,
        const double * directions,
        const double * magnitudes,
        double * fields)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {spins[3*idx], spins[3*idx+1], spins[3*idx+2]};
    const double x[3] = {directions[3*idx], directions[3*idx+1], directions[3*idx+2]};
    const double d = magnitudes[idx];

#pragma unroll
    for (auto n = 0; n < 3; ++n) {
      fields[3 * idx + n] = d * dot(x, s) * x[n];
    }
  }
}

__global__
void random_anisotropy_cuda_energy_kernel(
        const int num_spins,
        const double * spins,
        const double * directions,
        const double * magnitudes,
        double * energies)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {spins[3 * idx], spins[3 * idx + 1], spins[3 * idx + 2]};
    const double x[3] = {directions[3 * idx], directions[3 * idx + 1], directions[3 * idx + 2]};
    const double d = magnitudes[idx];

    energies[idx] = -d * pow2(dot(x, s));
  }
}

#endif //JAMS_RANDOM_ANISOTROPY_CUDA_KERNEL_H
