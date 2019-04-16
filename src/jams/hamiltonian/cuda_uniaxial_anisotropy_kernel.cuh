#include "jams/cuda/cuda_device_vector_ops.h"

__device__ double dot(const double3 &a, const double3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void cuda_uniaxial_energy_kernel(const int num_spins, const int num_coefficients, const unsigned * power,
  const double * magnitude, const double3 * axis, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double3 s = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    double energy = 0.0;

    for (auto n = 0; n < num_coefficients; ++n) {
      auto s_dot_a = dot(s, axis[num_coefficients * idx + n]);
      energy += (-magnitude[num_coefficients * idx + n] * pow(s_dot_a, power[num_coefficients * idx + n]));
    }

    dev_e[idx] =  energy;
  }
}

__global__ void cuda_uniaxial_field_kernel(const int num_spins, const int num_coefficients, const unsigned * power,
                                           const double * magnitude, const double * dev_axis, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double spin[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    double field[3] = {0.0, 0.0, 0.0};

    for (auto n = 0; n < num_coefficients; ++n) {
      const double axis[3] = {dev_axis[3 * (num_coefficients * idx + n) + 0], dev_axis[3 * (num_coefficients * idx + n) + 1],
                              dev_axis[3 * (num_coefficients * idx + n) + 2]};
      unsigned p = power[num_coefficients * idx + n];
      double s_dot_a = dot(spin, axis);
      double pre = magnitude[num_coefficients * idx + n] * p * pow(s_dot_a, p - 1);
      for (auto j = 0; j < 3; ++j) {
        field[j] += pre * axis[j];
      }
    }

    for (auto j = 0; j < 3; ++j) {
      dev_h[3 * idx + j] = field[j];
    }
  }
}
