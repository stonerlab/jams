#include "jams/cuda/cuda_device_vector_ops.h"

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
                                           const double * magnitude, const double * axis, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    double field[3] = {0.0, 0.0, 0.0};

    for (auto n = 0; n < num_coefficients; ++n) {
      double a[3] = {axis[3*(num_coefficients * idx + n)], axis[3*(num_coefficients * idx + n)+1], axis[3*(num_coefficients * idx + n)+2]};

      auto p = power[num_coefficients * idx + n];
      auto s_dot_a = s[0] * a[0] + s[1] * a[1] + s[2] * a[2];
      auto pre = magnitude[num_coefficients * idx + n] * p * pow(s_dot_a, p-1);
      for (auto j = 0; j < 3; ++j) {
        field[j] += pre * a[j];
      }
    }

    dev_h[3 * idx] = field[0];
    dev_h[3 * idx + 1] = field[1];
    dev_h[3 * idx + 2] = field[2];
  }
}
