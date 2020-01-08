#include "jams/cuda/cuda_device_vector_ops.h"

__device__ double dot(const double3 &a, const double3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void cuda_cubic_energy_kernel(const int num_spins, const int num_coefficients, const unsigned * power,
                                            const double * magnitude, const double3 * axis, const double * dev_s, double * dev_e) {
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < num_spins) {
        const double3 s = {dev_s[3 * idx], dev_s[3 * idx + 1], dev_s[3 * idx + 2]};
        double energy = 0.0;

        for (auto n = 0; n < num_coefficients; ++n) {
            auto s_dot_a = dot(s, axis[num_coefficients * idx + n]);
            auto s_dot_b = dot(s, axis[num_coefficients * idx + n + 1]);
            auto s_dot_c = dot(s, axis[num_coefficients * idx + n + 2]);

            if (power[num_coefficients * idx + n] == 1){
                energy += -magnitude[num_coefficients * idx + n] * (pow(s_dot_a, 2) * pow(s_dot_b, 2) + pow(s_dot_b, 2) * pow(s_dot_c, 2) + pow(s_dot_a, 2) * pow(s_dot_c, 2));
            }

            if (power[num_coefficients * idx + n] == 2){
                energy += -magnitude[num_coefficients * idx + n] * pow(s_dot_a, 2) * pow(s_dot_b, 2) * pow(s_dot_c, 2);
            }
        }
        dev_e[idx] = energy;
    }
}

__global__ void cuda_cubic_field_kernel(const int num_spins, const int num_coefficients, const unsigned * power,
                                           const double * magnitude, const double * axis, const double * dev_s, double * dev_h) {
    //Clearly idx is to do with which core on the GPU you're working with
    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < num_spins) {
        const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
        double field[3] = {0.0, 0.0, 0.0};

        for (auto n = 0; n < num_coefficients; ++n) {
            double a[3] = {axis[3*(num_coefficients * idx + n)], axis[3*(num_coefficients * idx + n) + 1], axis[3*(num_coefficients * idx + n) + 2]};
            double b[3] = {axis[3*(num_coefficients * idx + 1 + n)], axis[3*(num_coefficients * idx + 1 + n) + 1], axis[3*(num_coefficients * idx + 1 + n) + 2]};
            double c[3] = {axis[3*(num_coefficients * idx + 2 + n)], axis[3*(num_coefficients * idx + 2 + n) + 1], axis[3*(num_coefficients * idx + 2 + n) + 2]};

            auto p = power[num_coefficients * idx + n];
            auto s_dot_a = s[0] * a[0] + s[1] * a[1] + s[2] * a[2];
            auto s_dot_b = s[0] * b[0] + s[1] * b[1] + s[2] * b[3];
            auto s_dot_c = s[0] * c[0] + s[1] * c[1] + s[2] * c[2];

            if (p == 1) {
                auto pre = 2.0 * magnitude[num_coefficients * idx + n];
                for (auto j = 0; j < 3; ++j) {
                    field[j] += pre * (a[j] * s_dot_a * (s_dot_b + s_dot_c) + b[j] * s_dot_b * (s_dot_a + s_dot_c) + c[j] * s_dot_c * (s_dot_a + s_dot_b));
                }
            }

            if (p == 2) {
                auto pre = 2.0 * magnitude[num_coefficients * idx + n] * (s_dot_a + s_dot_b + s_dot_c);
                for (auto j = 0; j < 3; ++j) {
                    field[j] += pre * (s_dot_b * s_dot_c * a[j] + s_dot_a * s_dot_c * b[j] + s_dot_a * s_dot_b * c[j]);
                }
            }
        }

        dev_h[3 * idx] = field[0];
        dev_h[3 * idx + 1] = field[1];
        dev_h[3 * idx + 2] = field[2];
    }
}
