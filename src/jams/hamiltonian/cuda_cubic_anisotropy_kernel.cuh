#include "jams/cuda/cuda_device_vector_ops.h"

__global__ void cuda_cubic_energy_kernel(const int num_spins, const int num_coefficients, const unsigned * order,
                                         const double * magnitude, const double3 * axis1, const double3 * axis2, const double3 * axis3, const double * dev_s, double * dev_e) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double3 s = {dev_s[3 * idx], dev_s[3 * idx + 1], dev_s[3 * idx + 2]};
    double energy = 0.0;

    for (auto n = 0; n < num_coefficients; ++n) {
      auto s_dot_a = dot(s, axis1[num_coefficients * idx + n]);
      auto s_dot_b = dot(s, axis2[num_coefficients * idx + n]);
      auto s_dot_c = dot(s, axis3[num_coefficients * idx + n]);

      if (order[num_coefficients * idx + n] == 1){
        energy += -magnitude[num_coefficients * idx + n] * (pow(s_dot_a, 2) * pow(s_dot_b, 2) + pow(s_dot_b, 2) * pow(s_dot_c, 2) + pow(s_dot_c, 2) * pow(s_dot_a, 2));
      }

      if (order[num_coefficients * idx + n] == 2){
        energy += -magnitude[num_coefficients * idx + n] * pow(s_dot_a, 2) * pow(s_dot_b, 2) * pow(s_dot_c, 2);
      }
    }
    dev_e[idx] = energy;
  }
}

__global__ void cuda_cubic_field_kernel(const int num_spins, const int num_coefficients, const unsigned * order,
                                        const double * magnitude, const double * axis1, const double * axis2, const double * axis3, const double * dev_s, double * dev_h) {
  const int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num_spins) {
    const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
    double field[3] = {0.0, 0.0, 0.0};

    for (auto n = 0; n < num_coefficients; ++n) {
      double a[3] = {axis1[3*(num_coefficients * idx + n)], axis1[3*(num_coefficients * idx + n)+1], axis1[3*(num_coefficients * idx + n)+2]};
      double b[3] = {axis2[3*(num_coefficients * idx + n)], axis2[3*(num_coefficients * idx + n)+1], axis2[3*(num_coefficients * idx + n)+2]};
      double c[3] = {axis3[3*(num_coefficients * idx + n)], axis3[3*(num_coefficients * idx + n)+1], axis3[3*(num_coefficients * idx + n)+2]};


      auto s_dot_a = dot(a, s);
      auto s_dot_b = dot(b, s);
      auto s_dot_c = dot(c, s);

      auto pre = 2.0 * magnitude[num_coefficients * idx + n];

      if (order[num_coefficients * idx + n] == 1) {

        for (auto j = 0; j < 3; ++j) {
          field[j] += pre * (a[j] * s_dot_a * (pow(s_dot_b,2) + pow(s_dot_c,2)) + b[j] * s_dot_b * (pow(s_dot_a,2) + pow(s_dot_c,2)) + c[j] * s_dot_c * (pow(s_dot_a,2) + pow(s_dot_b,2)));
        }
      }

      if (order[num_coefficients * idx + n] == 2) {
        for (auto j = 0; j < 3; ++j) {
          field[j] += pre * (a[j] * s_dot_a * (pow(s_dot_b,2) * pow(s_dot_c,2)) + b[j] * s_dot_b * (pow(s_dot_a,2) * pow(s_dot_c,2)) + c[j] * s_dot_c * (pow(s_dot_a,2) * pow(s_dot_b,2)));
        }
      }
    }

    dev_h[3 * idx] = field[0];
    dev_h[3 * idx + 1] = field[1];
    dev_h[3 * idx + 2] = field[2];
  }
}
