#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_KERNEL_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_KERNEL_H
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
                const int n,
                const int num_freq,
                const int num_proc // 3*num_spins
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_proc) {
    double sum = 0.0;
    for (auto m = 0; m < (2*num_freq - 1); ++m) {
      const auto n_m = pbc(n-m, (2*num_freq - 1));
      auto rand_idx = num_proc*n_m + idx;
      sum += filter[m] * rands[rand_idx];
    }
    noise_[idx] = sum;
  }
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_KERNEL_H
