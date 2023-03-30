//
// Created by Sean Stansill [ll14s26s] on 30/03/2023.
//
#include "jams/cuda/cuda_device_vector_ops.h"

__global__ void cuda_uniaxial_generalised_energy_kernel(const int num_spins,
                const double * magnitude, const double3 * axis1, const double3 * axis2, const double3 * axis3,
                const double * dev_s, double * dev_e, const double a1, const double a2, const double a3, const double a4,
                const double a5, const double a6) {

    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < num_spins) {
        const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
        const double u[3] = {axis1[3*idx], axis1[3*idx+1], axis1[3*idx+2]};
        const double v[3] = {axis2[3*idx], axis2[3*idx+1], axis2[3*idx+2]};
        const double w[3] = {axis3[3*idx], axis3[3*idx+1], axis3[3*idx+2]};
        const double su = dot(s, u);
        const double sv = dot(s, v);
        const double sw = dot(s, w);
        const double su2 = pow(su,2);
        const double sv2 = pow(sv,2);
        const double sw2 = pow(sw,2);

        dev_e[idx] = -magnitude[idx] * ( a1*sw2 + a2*pow(sw2, 2) +
                    a3*sw*sv*( 3*su2 - sv2 ) + a4*(pow(su2, 2) + pow(sv2, 2)) +
                    a5*(su2*sv2) + a6*(pow(su2, 2) - 6*su2*sv2 + pow(sv2, 2)) );
    }
}

__global__ void cuda_uniaxial_generalised_field_kernel(const int num_spins,
                       const double * magnitude, const double3 * axis1, const double3 * axis2, const double3 * axis3,
                       const double * dev_s, double * dev_h, const double a1, const double a2, const double a3,
                       const double a4, const double a5, const double a6) {

    const int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < num_spins) {
        const double s[3] = {dev_s[3*idx], dev_s[3*idx+1], dev_s[3*idx+2]};
        const double u[3] = {axis1[3*idx], axis1[3*idx+1], axis1[3*idx+2]};
        const double v[3] = {axis2[3*idx], axis2[3*idx+1], axis2[3*idx+2]};
        const double w[3] = {axis3[3*idx], axis3[3*idx+1], axis3[3*idx+2]};
        const double su = dot(s, u);
        const double sv = dot(s, v);
        const double sw = dot(s, w);
        const double su2 = pow(su,2);
        const double sv2 = pow(sv,2);
        const double sw2 = pow(sw,2);

        dev_h[3 * idx] = magnitude[idx] * (
                w[0]*(a1*2*sw + a2*4*pow(sw, 3) + a3*(sv*(3*su2-sv2))) +
                v[0]*( a3*(sv*(3*su2-sv2-2*sv*sw)) + a4*4*pow(sv, 3) + a5*2*su2*sv + a6*(sv*(4*sv2-6*su2)) ) +
                u[0]*( a3*(6*su*sv*sw) + a4*4*pow(su, 3) + a5*2*su*sv2 + a6*su*(4*su2-6*sv2) ));

        dev_h[3 * idx + 1] = magnitude[idx] * (
                w[1]*(a1*2*sw + a2*4*pow(sw, 3) + a3*(sv*(3*su2-sv2))) +
                v[1]*( a3*(sv*(3*su2-sv2-2*sv*sw)) + a4*4*pow(sv, 3) + a5*2*su2*sv + a6*(sv*(4*sv2-6*su2)) ) +
                u[1]*( a3*(6*su*sv*sw) + a4*4*pow(su, 3) + a5*2*su*sv2 + a6*su*(4*su2-6*sv2) ));

        dev_h[3 * idx + 2] = magnitude[idx] * (
                w[2]*(a1*2*sw + a2*4*pow(sw, 3) + a3*(sv*(3*su2-sv2))) +
                v[2]*( a3*(sv*(3*su2-sv2-2*sv*sw)) + a4*4*pow(sv, 3) + a5*2*su2*sv + a6*(sv*(4*sv2-6*su2)) ) +
                u[2]*( a3*(6*su*sv*sw) + a4*4*pow(su, 3) + a5*2*su*sv2 + a6*su*(4*su2-6*sv2) ));
    }
}
