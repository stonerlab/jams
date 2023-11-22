#include "jams/cuda/cuda_device_vector_ops.h"

__global__ void cuda_cubic_energy_kernel(const int num_spins, const unsigned * order,
                                         const double * magnitude, const double * axis1, const double * axis2, const double * axis3, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {dev_s[3 * idx], dev_s[3 * idx + 1], dev_s[3 * idx + 2]};
    double energy = 0.0;

    const double u[3] = {axis1[3*idx], axis1[3*idx+1], axis1[3*idx+2]};
    const double v[3] = {axis2[3*idx], axis2[3*idx+1], axis2[3*idx+2]};
    const double w[3] = {axis3[3*idx], axis3[3*idx+1], axis3[3*idx+2]};


    auto su2 = pow2(dot(s, u));
    auto sv2 = pow2(dot(s, v));
    auto sw2 = pow2(dot(s, w));

  if (order[idx] == 1){
    energy += -magnitude[idx] * (su2 * sv2 + sv2 * sw2 + sw2 * su2);
  }

  if (order[idx] == 2){
    energy += -magnitude[idx] * su2 * sv2 * sw2;
  }

    dev_e[idx] = energy;
  }
}

__global__ void cuda_cubic_field_kernel(const int num_spins, const unsigned * order,
                                        const double * magnitude, const double * axis1, const double * axis2, const double * axis3, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    double field[3] = {0.0, 0.0, 0.0};

  const double u[3] = {axis1[3*idx], axis1[3*idx+1], axis1[3*idx+2]};
  const double v[3] = {axis2[3*idx], axis2[3*idx+1], axis2[3*idx+2]};
  const double w[3] = {axis3[3*idx], axis3[3*idx+1], axis3[3*idx+2]};


  auto su = dot(s, u);
  auto sv = dot(s, v);
  auto sw = dot(s, w);

  auto pre = 2.0 * magnitude[idx];

  if (order[idx] == 1) {

    for (auto j = 0; j < 3; ++j) {
      field[j] += pre * ( u[j] * su * (pow2(sv) + pow2(sw)) + v[j] * sv * (pow2(sw) + pow2(su)) + w[j] * sw * (pow2(su) + pow2(sv)) );
    }
  }

  if (order[idx] == 2) {
    for (auto j = 0; j < 3; ++j) {
      field[j] += pre * ( u[j] * su * pow2(sv) * pow2(sw) + v[j] * sv * pow2(sw) * pow2(su) + w[j] * sw * pow2(su) * pow2(sv) );
    }
  }

    dev_h[3 * idx] = field[0];
    dev_h[3 * idx + 1] = field[1];
    dev_h[3 * idx + 2] = field[2];
  }
}
