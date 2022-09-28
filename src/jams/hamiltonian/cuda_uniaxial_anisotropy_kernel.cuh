#include "jams/cuda/cuda_device_vector_ops.h"

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

__global__ void cuda_uniaxial_helicity_energy_kernel(const int num_spins, const int power,
  const double * magnitude, const double * axis, const double * dev_s, double * dev_dU) {
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < num_spins) {
        const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
        const double a[3] = {axis[3*idx], axis[3*idx+1], axis[3*idx+2]};

        const double cross[3] = {cross_product_x(s, a), cross_product_y(s, a), cross_product_z(s, a)};
        const double cross_sq = dot(cross, cross);

        dev_dU[idx] = magnitude[idx] * power * (pow(dot(s, a), power) - (power - 1)*cross_sq*pow(dot(s, a), power-2) );
    }
}
__global__ void cuda_uniaxial_entropy_kernel(const int num_spins, const int power,
                                                     const double * magnitude, const double * axis, const double * dev_s, double * dev_TS) {
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < num_spins) {
        const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
        const double a[3] = {axis[3*idx], axis[3*idx+1], axis[3*idx+2]};

        const double cross[3] = {cross_product_x(s, a), cross_product_y(s, a), cross_product_z(s, a)};
        const double cross_sq = dot(cross, cross);

        dev_TS[idx] = pow2(magnitude[idx] * power) * pow(dot(s, a), 2*power-2) * cross_sq;
    }
}
