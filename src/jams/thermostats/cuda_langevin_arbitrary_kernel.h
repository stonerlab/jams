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
    double sum = filter[0]*rands[j];

    for (auto k = 1; k < num_trunc + 1; ++k) {
      const auto j_minus_k = pbc(j-k, (2*num_trunc + 1));
      const auto j_plus_k  = pbc(j+k, (2*num_trunc + 1));

      sum += filter[k] * (rands[num_proc*j_plus_k + idx] + rands[num_proc*j_minus_k + idx]);
    }

    noise_[idx] = sum;
  }
}

__global__ void lorentzian_memory_cuda_kernel(
    double *w_data,
    double *v_data,
    const double * s_data,
    const double * gyro_data,
    const double omega,
    const double gamma,
    const double A,
    const double dt,
    const unsigned num_spins
    ) {


  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int idy = threadIdx.y;

  if (idx < num_spins && idy < 3) {

    double s = s_data[3*idx + idy];
    double gyro = gyro_data[idx];

    double y[2] = {
        w_data[3*idx + idy],  // w -> y[0]
        v_data[3*idx + idy]}; // v -> y[1]

    nvstd::function<void(double[2], const double[2])> ode_system = [&](double out[2], const double in[2]) {
      const double w = in[0];
      const double v = in[1];

      out[0] = -omega * omega * v - gamma * w - A * gyro * s;
      out[1] = w;
    };


    rk4<2>(ode_system, y, dt);

    w_data[3*idx + idy] = y[0];
    v_data[3*idx + idy] = y[1];
  }
}

#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_KERNEL_H
