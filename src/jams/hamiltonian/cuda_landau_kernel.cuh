#include "jams/cuda/cuda_device_vector_ops.h"

__global__
void cuda_landau_field_kernel(
    const int num_spins,
    const double * spins,
    const double *A,
    const double *B,
    const double *C,
    double * fields)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {spins[3*idx], spins[3*idx+1], spins[3*idx+2]};
    double s2 = dot(s, s);

    double h[3] = {0, 0, 0};

    for (auto n = 0; n < 3; ++n) {
      h[n] += -2.0 * A[idx] * s[n];
    }

    for (auto n = 0; n < 3; ++n) {
      h[n] += -4.0 * B[idx] * s[n] * s2;
    }


    for (auto n = 0; n < 3; ++n) {
      h[n] += -6.0 * C[idx] * s[n] * s2 * s2;
    }

    for (auto n = 0; n < 3; ++n) {
      fields[3 * idx + n] = h[n];
    }
  }
}

__global__
void cuda_landau_energy_kernel(
    const int num_spins,
    const double * spins,
    const double *A,
    const double *B,
    const double *C,
    double * energies)
{
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {spins[3 * idx], spins[3 * idx + 1], spins[3 * idx + 2]};

    double s2 = dot(s, s);

    energies[idx] = A[idx] * s2 + B[idx] * s2 * s2 + C[idx] * s2 * s2 * s2;
  }
}