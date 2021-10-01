#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_KERNEL_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_KERNEL_H

#include "jams/cuda/cuda_device_rk4.cuh"

__host__ __device__ inline int pbc(const int i, const int size) {
  return (i + size) % size;
}

// all threads should pull the same n-m at the same time, i.e.
// a given n-m should be contiguous for all num_proc

__global__ void arbitrary_stochastic_process_cuda_kernel
        (
                double *noise_,
                const double *filter,
                const double *rands,
                const int j, // pointer to central random number
                const int num_trunc,
                const int num_proc // 3*num_spins
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_proc) {
    double sum = filter[0]*rands[num_proc*j + idx];

    for (auto k = 1; k < num_trunc; ++k) {
      const auto j_minus_k = pbc(j-k, (2*num_trunc + 1));
      const auto j_plus_k  = pbc(j+k, (2*num_trunc + 1));

      sum += filter[k] * (rands[num_proc*j_plus_k + idx] + rands[num_proc*j_minus_k + idx]);
    }

    noise_[idx] = sum;
  }
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_KERNEL_H
