#include "jams/cuda/cuda_device_vector_ops.h"

__device__ double dot(const double3 &a, const double3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void cuda_uniaxial_energy_kernel(const int num_spins, const int power,
  const double * magnitude, const double * axis, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    const double a[3] = {axis[3*idx], axis[3*idx+1], axis[3*idx+2]};

    dev_e[idx] = -magnitude[idx] * pow(dot(s, a), power);
  }
}

__global__ void cuda_uniaxial_field_kernel(const int num_spins, const int power,
                                           const double * magnitude, const double * axis, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    const double a[3] = {axis[3*idx], axis[3*idx+1], axis[3*idx+2]};
    const auto pre = magnitude[idx] * power * pow(dot(s, a), power-1);

    dev_h[3 * idx]     = pre * a[0];
    dev_h[3 * idx + 1] = pre * a[1];
    dev_h[3 * idx + 2] = pre * a[2];
  }
}
