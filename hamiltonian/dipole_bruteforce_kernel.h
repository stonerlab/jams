#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H

#include "core/cuda_defs.h"

__constant__ float dev_super_unit_cell[3][3];
__constant__ float dev_super_unit_cell_inv[3][3];
__constant__ bool   dev_super_cell_pbc[3];
__constant__ double dev_dipole_prefactor;
__constant__ float dev_r_cutoff;


__device__ inline void displacement(
    const float ri[3],
    const float rj[3],
    float dr[3]
) {
    float dr_frac[3];

    // transform into fractional coordinates
    dr_frac[0] = dev_super_unit_cell_inv[0][0] * (ri[0] - rj[0])
          + dev_super_unit_cell_inv[0][1] * (ri[1] - rj[1])
          + dev_super_unit_cell_inv[0][2] * (ri[2] - rj[2]);

    dr_frac[1] = dev_super_unit_cell_inv[1][0] * (ri[0] - rj[0])
          + dev_super_unit_cell_inv[1][1] * (ri[1] - rj[1])
          + dev_super_unit_cell_inv[1][2] * (ri[2] - rj[2]);

    dr_frac[2] = dev_super_unit_cell_inv[2][0] * (ri[0] - rj[0])
          + dev_super_unit_cell_inv[2][1] * (ri[1] - rj[1])
          + dev_super_unit_cell_inv[2][2] * (ri[2] - rj[2]);

    // apply boundary conditions
    #pragma unroll
    for (int n = 0; n < 3; ++n) {
      if (dev_super_cell_pbc[n]) {
        // W. Smith, CCP5 Information Quarterly for Computer Simulation of Condensed Phases (1989).
        dr_frac[n] = dr_frac[n] - trunc(2.0 * dr_frac[n]);
      }
    }

    // transform back to cartesian space
    dr[0] = dev_super_unit_cell[0][0] * dr_frac[0]
          + dev_super_unit_cell[0][1] * dr_frac[1]
          + dev_super_unit_cell[0][2] * dr_frac[2];

    dr[1] = dev_super_unit_cell[1][0] * dr_frac[0]
          + dev_super_unit_cell[1][1] * dr_frac[1]
          + dev_super_unit_cell[1][2] * dr_frac[2];

    dr[2] = dev_super_unit_cell[2][0] * dr_frac[0]
          + dev_super_unit_cell[2][1] * dr_frac[1]
          + dev_super_unit_cell[2][2] * dr_frac[2];
}

__global__ void dipole_bruteforce_kernel
(
    const double * s_dev,
    const float * r_dev,
    const double * mus_dev,
    const unsigned int num_spins,
    double * h_dev
)
{
    const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int j, n;

    double h[3] = {0.0, 0.0, 0.0};
    double sj[3];
    float ri[3], rj[3], r_ij[3], r_abs;
    double sj_dot_rhat, w0;

    if (idx < num_spins) {
        // constant is kB * mu_0 / 4 pi
        const double pre = mus_dev[idx] * dev_dipole_prefactor;

        #pragma unroll
        for (n = 0; n < 3; ++n) {
            ri[n] = r_dev[3*idx + n];
        }


        for (j = 0; j < num_spins; ++j) {
            if (idx == j) continue;

            #pragma unroll
            for (n = 0; n < 3; ++n) {
                rj[n] = r_dev[3*j + n];
            }

            displacement(ri, rj, r_ij);

            r_abs = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2];

            if (r_abs > dev_r_cutoff * dev_r_cutoff) continue;

            // r_abs = 1.0 / sqrt(r_abs);

            r_abs = rsqrtf(r_abs);

            w0 = pre * mus_dev[j] * (r_abs * r_abs * r_abs);

            #pragma unroll
            for (n = 0; n < 3; ++n) {
                sj[n] = s_dev[3*j + n];
            }

            sj_dot_rhat = 3.0 * (sj[0] * r_ij[0] + sj[1] * r_ij[1] + sj[2] * r_ij[2]) * r_abs* r_abs;

            // accumulate values
            #pragma unroll
            for (n = 0; n < 3; ++n) {
                h[n] += (r_ij[n] * sj_dot_rhat  - sj[n]) * w0;
            }
        }

        __syncthreads();
        // write to global memory
        #pragma unroll
        for (n = 0; n < 3; ++n) {
            h_dev[3*idx + n] = h[n];
        }
    }
}

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H
