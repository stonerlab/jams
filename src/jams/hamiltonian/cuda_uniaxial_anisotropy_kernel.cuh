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
                                           const double * magnitude, const double3 * axis, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double3 s = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    double3 field = {0.0, 0.0, 0.0};

    for (auto n = 0; n < num_coefficients; ++n) {
      auto a = axis[num_coefficients * idx + n];
      auto p = power[num_coefficients * idx + n];
      auto s_dot_a = dot(s, a);
      auto pre = magnitude[num_coefficients * idx + n] * p * pow(s_dot_a, p-1);
      field.x += pre * a.x;
      field.y += pre * a.y;
      field.z += pre * a.z;
    }

    dev_h[3 * idx] = field.x;
    dev_h[3 * idx + 1] = field.y;
    dev_h[3 * idx + 2] = field.z;
  }
}
