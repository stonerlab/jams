#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H

#include "core/cuda_defs.h"

__constant__ float dev_super_unit_cell[3][3];
__constant__ float dev_super_unit_cell_inv[3][3];
__constant__ bool   dev_super_cell_pbc[3];
__constant__ float dev_dipole_prefactor;
__constant__ float dev_r_cutoff;

constexpr int shared_size = 256;

__device__ inline void displacement(
    const float ri[3],
    const float rj[3],
    float dr[3]
) {
    float dr_frac[3];
    const float rij[3] 
      = {ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]};

    // transform into fractional coordinates
    dr_frac[0] = dev_super_unit_cell_inv[0][0] * (rij[0])
               + dev_super_unit_cell_inv[0][1] * (rij[1])
               + dev_super_unit_cell_inv[0][2] * (rij[2]);

    dr_frac[1] = dev_super_unit_cell_inv[1][0] * (rij[0])
               + dev_super_unit_cell_inv[1][1] * (rij[1])
               + dev_super_unit_cell_inv[1][2] * (rij[2]);

    dr_frac[2] = dev_super_unit_cell_inv[2][0] * (rij[0])
               + dev_super_unit_cell_inv[2][1] * (rij[1])
               + dev_super_unit_cell_inv[2][2] * (rij[2]);

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

// __global__ void dipole_bruteforce_kernel
// (
//     const double * s_dev,
//     cudaTextureObject_t r_tex,
//     const float * mus_dev,
//     const unsigned int num_spins,
//     double * h_dev
// )
// {
//     const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
//     unsigned int n;

//     double h[3] = {0.0, 0.0, 0.0};

//     float sj[3], ri[3], rj[3], r_ij[3], r_abs, w0, sj_dot_rhat;

//     if (idx < num_spins) {
//         // constant is kB * mu_0 / 4 pi
//         const float pre = mus_dev[idx] * dev_dipole_prefactor;

//         for (n = 0; n < 3; ++n) {
//             ri[n] = tex1Dfetch<float>(r_tex, 3*idx + n);
//         }

//         for (int j = 0; j < num_spins; ++j) {
//           // __syncthreads();

//           for (n = 0; n < 3; ++n) {
//               rj[n] = tex1Dfetch<float>(r_tex, 3*j + n);
//           }

//           for (n = 0; n < 3; ++n) {
//               sj[n] = s_dev[3*j + n];
//           }

//           displacement(ri, rj, r_ij);

//           r_abs = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2];

//           if (j == idx || r_abs > (dev_r_cutoff * dev_r_cutoff)) continue;

//           r_abs = rsqrtf(r_abs);

//           w0 = pre * mus_dev[j] * (r_abs * r_abs * r_abs);

//           sj_dot_rhat = 3.0f * (sj[0] * r_ij[0] + sj[1] * r_ij[1] + sj[2] * r_ij[2]) * r_abs * r_abs;

//           // accumulate values
//           for (n = 0; n < 3; ++n) {
//               h[n] += (r_ij[n] * sj_dot_rhat  - sj[n]) * w0;
//           }
//         }

//         // write to global memory
//         #pragma unroll
//         for (n = 0; n < 3; ++n) {
//             h_dev[3*idx + n] = h[n];
//         }
//     }
// }

__global__ void dipole_bruteforce_kernel
(
    const double * s_dev,
    const float * r_dev,
    const float * mus_dev,
    const unsigned int num_spins,
    double * h_dev
)
{
    const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int tx = threadIdx.x;

    __shared__ float rj[shared_size][3];
    __shared__ float sj[shared_size][3];
    __shared__ float mus[shared_size];

    if (idx < num_spins) {
        // constant is kB * mu_0 / 4 pi
        const float pre = mus_dev[idx] * dev_dipole_prefactor;

        const float ri[3] = {r_dev[3*idx], r_dev[3*idx + 1], r_dev[3*idx + 2]};

        const unsigned int num_blocks = (num_spins + shared_size - 1) / shared_size;

        double h[3] = {0.0, 0.0, 0.0};
        for (unsigned int b = 0; b < num_blocks; ++b) {
          
          const unsigned int shared_idx = tx + shared_size * b;

          if (shared_idx < num_spins) {
            
            mus[tx] = mus_dev[shared_idx];

            for (unsigned int n = 0; n < 3; ++n) {
                rj[tx][n] = r_dev[3*shared_idx + n];
            }
            for (unsigned int n = 0; n < 3; ++n) {
                sj[tx][n] = s_dev[3*shared_idx + n];
            }

            __syncthreads();
            for (unsigned int t = 0; t < shared_size; ++t) {

              float r_ij[3];
              displacement(ri, rj[t], r_ij);

              float r_abs = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2];

              if (r_abs > dev_r_cutoff * dev_r_cutoff || idx == (shared_size * b + t)) continue;

              r_abs = rsqrtf(r_abs);

              const float w0 = pre * mus[t] * (r_abs * r_abs * r_abs);

              const float sj_dot_rhat = 3.0f * (sj[t][0] * r_ij[0] + sj[t][1] * r_ij[1] + sj[t][2] * r_ij[2]) * r_abs * r_abs;

              // accumulate values
              #pragma unroll
              for (unsigned int n = 0; n < 3; ++n) {
                  h[n] += (r_ij[n] * sj_dot_rhat  - sj[t][n]) * w0;
              }
              }
            }
          }
        // write to global memory
        #pragma unroll
        for (unsigned int n = 0; n < 3; ++n) {
            h_dev[3*idx + n] = h[n];
        }
    }
}

#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H
