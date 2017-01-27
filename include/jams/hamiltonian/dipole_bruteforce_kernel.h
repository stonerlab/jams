#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H

#include "jams/core/cuda_defs.h"

__constant__ float dev_super_unit_cell[3][3];
__constant__ float dev_super_unit_cell_inv[3][3];
__constant__ bool   dev_super_cell_pbc[3];
__constant__ float dev_dipole_prefactor;
__constant__ float dev_r_cutoff;

constexpr int block_size = 128;

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

__device__ void field_element(const float sj[3], const float r_ij[3], const float pre, const float r_abs_sq, float h[3]) {

  const float d_r_abs = rsqrtf(r_abs_sq);

  const float sj_dot_rhat = 3.0f * (sj[0] * r_ij[0] + sj[1] * r_ij[1] + sj[2] * r_ij[2]);

  const float w0 = pre * d_r_abs * d_r_abs * d_r_abs * d_r_abs * d_r_abs;

  h[0] = w0 * (r_ij[0] * sj_dot_rhat  - r_abs_sq * sj[0]);
  h[1] = w0 * (r_ij[1] * sj_dot_rhat  - r_abs_sq * sj[1]);
  h[2] = w0 * (r_ij[2] * sj_dot_rhat  - r_abs_sq * sj[2]);
}


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
    unsigned int n;

    double h[3] = {0.0, 0.0, 0.0};

    float sj[3], ri[3], rj[3], r_ij[3], r_abs, sj_dot_rhat;

    if (idx < num_spins) {
        // constant is kB * mu_0 / 4 pi
        const float pre = mus_dev[idx] * dev_dipole_prefactor;

        for (n = 0; n < 3; ++n) {
            ri[n] = r_dev[3*idx + n];
        }

        for (int j = 0; j < num_spins; ++j) {

          for (n = 0; n < 3; ++n) {
              rj[n] = r_dev[3*j + n];
          }

          for (n = 0; n < 3; ++n) {
              sj[n] = s_dev[3*j + n];
          }

          displacement(ri, rj, r_ij);

          r_abs = sqrtf(r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]);

          if (j == idx || r_abs > dev_r_cutoff) continue;

          sj_dot_rhat = (sj[0] * r_ij[0] + sj[1] * r_ij[1] + sj[2] * r_ij[2]);

          // accumulate values
          for (n = 0; n < 3; ++n) {
              h[n] += pre * mus_dev[j] * (3.0f * r_ij[n] * sj_dot_rhat  - r_abs * r_abs * sj[n]) / (r_abs * r_abs * r_abs * r_abs * r_abs);
          }
        }

        // write to global memory
        #pragma unroll
        for (n = 0; n < 3; ++n) {
            h_dev[3*idx + n] = h[n];
        }
    }
}


__global__ void dipole_bruteforce_pipeline4_kernel
(
    const double * __restrict__ s_dev,
    const float * __restrict__ r_dev,
    const float * __restrict__ mus_dev,
    const unsigned int num_spins,
    double * __restrict__ h_dev
)
{
    const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int tx = threadIdx.x; 

    __shared__ float rj[block_size][3];
    __shared__ float sj[block_size][3];
    __shared__ float mus_pre[block_size];

    float pre, ri[3];

    if (idx < num_spins) {
      // constant is kB * mu_0 / 4 pi
      pre = mus_dev[idx] * dev_dipole_prefactor;

      for (int n = 0; n < 3; ++n) {
          ri[n] = r_dev[3*idx + n];
      }
    }

    const unsigned int num_blocks = (num_spins + block_size - 1) / block_size;

    double h[3] = {0.0, 0.0, 0.0};
    for (unsigned int b = 0; b < num_blocks; ++b) {

      // for every block load up the shared memory with data
      
      const unsigned int shared_idx = tx + block_size * b;
      
      // sync to make sure we aren't using a dirty cache by other threads
      __syncthreads();

      // only load data which exists
      if (shared_idx < num_spins) {
        mus_pre[tx] = pre * mus_dev[shared_idx];

        for (unsigned int n = 0; n < 3; ++n) {
            rj[tx][n] = r_dev[3*shared_idx + n];
        }

        for (unsigned int n = 0; n < 3; ++n) {
            sj[tx][n] = s_dev[3*shared_idx + n];
        }
      }
      // sync to make sure the shared data is filled from all block threads
      __syncthreads();

      if (idx < num_spins) {

          //now process the loaded data with a 4x software pipeline
          // AKA - do shit loads of maths to hide memory latency
          for (unsigned int t = 0; t < block_size/4; ++t) {

            const int t1 = 4*t;
            const int t2 = 4*t+1;
            const int t3 = 4*t+2;
            const int t4 = 4*t+3;

            const int j1 = (block_size * b + t1);
            const int j2 = (block_size * b + t2);
            const int j3 = (block_size * b + t3);
            const int j4 = (block_size * b + t4);

            float r_ij1[3], r_ij2[3], r_ij3[3], r_ij4[3];


            const float pre1 = mus_pre[t1];
            const float pre2 = mus_pre[t2];
            const float pre3 = mus_pre[t3];
            const float pre4 = mus_pre[t4];


            displacement(ri, rj[t1], r_ij1);
            const float r_abs1 = (r_ij1[0] * r_ij1[0] + r_ij1[1] * r_ij1[1] + r_ij1[2] * r_ij1[2]);

            displacement(ri, rj[t2], r_ij2);
            const float r_abs2 = (r_ij2[0] * r_ij2[0] + r_ij2[1] * r_ij2[1] + r_ij2[2] * r_ij2[2]);

            displacement(ri, rj[t3], r_ij3);
            const float r_abs3 = (r_ij3[0] * r_ij3[0] + r_ij3[1] * r_ij3[1] + r_ij3[2] * r_ij3[2]);

            displacement(ri, rj[t4], r_ij4);
            const float r_abs4 = (r_ij4[0] * r_ij4[0] + r_ij4[1] * r_ij4[1] + r_ij4[2] * r_ij4[2]);


            float h1[3] = {0.0, 0.0, 0.0};
            float h2[3] = {0.0, 0.0, 0.0};
            float h3[3] = {0.0, 0.0, 0.0};
            float h4[3] = {0.0, 0.0, 0.0};

            if (r_abs1 <= (dev_r_cutoff * dev_r_cutoff) && idx != j1 && j1 < num_spins) {
              field_element(sj[t1], r_ij1, pre1, r_abs1, h1);
            }

            if (r_abs2 <= (dev_r_cutoff * dev_r_cutoff) && idx != j2 && j2 < num_spins) {
              field_element(sj[t2], r_ij2, pre2, r_abs2, h2);
            }

            if (r_abs3 <= (dev_r_cutoff * dev_r_cutoff) && idx != j3 && j3 < num_spins) {
              field_element(sj[t3], r_ij3, pre3, r_abs3, h3);
            }

            if (r_abs4 <= (dev_r_cutoff * dev_r_cutoff) && idx != j4 && j4 < num_spins) {
              field_element(sj[t4], r_ij4, pre4, r_abs4, h4);
            }

            h[0] = h[0] + h1[0] + h2[0] + h3[0] + h4[0];
            h[1] = h[1] + h1[1] + h2[1] + h3[1] + h4[1];
            h[2] = h[2] + h1[2] + h2[2] + h3[2] + h4[2];
          }
        }
      } // block loop

  if (idx < num_spins) {
    // write to global memory
    #pragma unroll
    for (unsigned int n = 0; n < 3; ++n) {
        h_dev[3*idx + n] = h[n];
    }
  }
}

// __global__ void dipole_bruteforce_pipeline4_kernel
// (
//     const double * __restrict__ s_dev,
//     const float * __restrict__ r_dev,
//     const float * __restrict__ mus_dev,
//     const unsigned int num_spins,
//     double * __restrict__ h_dev
// )
// {
//     const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
//     const unsigned int tx = threadIdx.x; 

//     __shared__ float rj[block_size][3];
//     __shared__ float sj[block_size][3];
//     __shared__ float mus_pre[block_size];

//     float pre, ri[3];

//     if (idx < num_spins) {
//       // constant is kB * mu_0 / 4 pi
//       pre = mus_dev[idx] * dev_dipole_prefactor;

//       for (int n = 0; n < 3; ++n) {
//           ri[n] = r_dev[3*idx + n];
//       }
//     }

//     const unsigned int num_blocks = (num_spins + block_size - 1) / block_size;

//     double h[3] = {0.0, 0.0, 0.0};
//     for (unsigned int b = 0; b < num_blocks; ++b) {

//       // for every block load up the shared memory with data
      
//       const unsigned int shared_idx = tx + block_size * b;
      
//       // sync to make sure we aren't using a dirty cache by other threads
//       __syncthreads();

//       // only load data which exists
//       if (shared_idx < num_spins) {
//         mus_pre[tx] = pre * mus_dev[shared_idx];

//         for (unsigned int n = 0; n < 3; ++n) {
//             rj[tx][n] = r_dev[3*shared_idx + n];
//         }

//         for (unsigned int n = 0; n < 3; ++n) {
//             sj[tx][n] = s_dev[3*shared_idx + n];
//         }
//       }
//       // sync to make sure the shared data is filled from all block threads
//       __syncthreads();

//       if (idx < num_spins) {

//           //now process the loaded data with a 4x software pipeline
//           for (unsigned int t = 0; t < block_size/4; ++t) {

//             const int t1 = 4*t;
//             const int t2 = 4*t+1;
//             const int t3 = 4*t+2;
//             const int t4 = 4*t+3;

//             const int j1 = (block_size * b + t1);
//             const int j2 = (block_size * b + t2);
//             const int j3 = (block_size * b + t3);
//             const int j4 = (block_size * b + t4);

//             float r_ij1[3], r_ij2[3], r_ij3[3], r_ij4[3];


//             const float pre1 = mus_pre[t1];
//             const float pre2 = mus_pre[t2];
//             const float pre3 = mus_pre[t3];
//             const float pre4 = mus_pre[t4];


//             displacement(ri, rj[t1], r_ij1);
//             const float r_abs1 = (r_ij1[0] * r_ij1[0] + r_ij1[1] * r_ij1[1] + r_ij1[2] * r_ij1[2]);

//             displacement(ri, rj[t2], r_ij2);
//             const float r_abs2 = (r_ij2[0] * r_ij2[0] + r_ij2[1] * r_ij2[1] + r_ij2[2] * r_ij2[2]);

//             displacement(ri, rj[t3], r_ij3);
//             const float r_abs3 = (r_ij3[0] * r_ij3[0] + r_ij3[1] * r_ij3[1] + r_ij3[2] * r_ij3[2]);

//             displacement(ri, rj[t4], r_ij4);
//             const float r_abs4 = (r_ij4[0] * r_ij4[0] + r_ij4[1] * r_ij4[1] + r_ij4[2] * r_ij4[2]);


//             float h1[3] = {0.0, 0.0, 0.0};
//             float h2[3] = {0.0, 0.0, 0.0};
//             float h3[3] = {0.0, 0.0, 0.0};
//             float h4[3] = {0.0, 0.0, 0.0};

//             if (r_abs1 <= (dev_r_cutoff * dev_r_cutoff) && idx != j1 && j1 < num_spins) {
//               field_element(sj[t1], r_ij1, pre1, r_abs1, h1);
//             }

//             if (r_abs2 <= (dev_r_cutoff * dev_r_cutoff) && idx != j2 && j2 < num_spins) {
//               field_element(sj[t2], r_ij2, pre2, r_abs2, h2);
//             }

//             if (r_abs3 <= (dev_r_cutoff * dev_r_cutoff) && idx != j3 && j3 < num_spins) {
//               field_element(sj[t3], r_ij3, pre3, r_abs3, h3);
//             }

//             if (r_abs4 <= (dev_r_cutoff * dev_r_cutoff) && idx != j4 && j4 < num_spins) {
//               field_element(sj[t4], r_ij4, pre4, r_abs4, h4);
//             }

//             h[0] = h[0] + h1[0] + h2[0] + h3[0] + h4[0];
//             h[1] = h[1] + h1[1] + h2[1] + h3[1] + h4[1];
//             h[2] = h[2] + h1[2] + h2[2] + h3[2] + h4[2];
//           }
//         }
//       } // block loop

//   if (idx < num_spins) {
//     // write to global memory
//     #pragma unroll
//     for (unsigned int n = 0; n < 3; ++n) {
//         h_dev[3*idx + n] = h[n];
//     }
//   }
// }

__global__ void dipole_bruteforce_sharemem_kernel
(
    const double * __restrict__ s_dev,
    const float * __restrict__ r_dev,
    const float * __restrict__ mus_dev,
    const unsigned int num_spins,
    double * __restrict__ h_dev
)
{
    const unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int tx = threadIdx.x; 

    __shared__ float rj[block_size][3];
    __shared__ float sj[block_size][3];
    __shared__ float mus_pre[block_size];

    float pre, ri[3];

    if (idx < num_spins) {
      // constant is kB * mu_0 / 4 pi
      pre = mus_dev[idx] * dev_dipole_prefactor;

      for (int n = 0; n < 3; ++n) {
          ri[n] = r_dev[3*idx + n];
      }
    }

    const unsigned int num_blocks = (num_spins + block_size - 1) / block_size;

    double h[3] = {0.0, 0.0, 0.0};
    for (unsigned int b = 0; b < num_blocks; ++b) {

      // for every block load up the shared memory with data
      
      const unsigned int shared_idx = tx + block_size * b;
      
      // sync to make sure we aren't using a dirty cache by other threads
      __syncthreads();

      // only load data which exists
      if (shared_idx < num_spins) {
        mus_pre[tx] = pre * mus_dev[shared_idx];

        for (unsigned int n = 0; n < 3; ++n) {
            rj[tx][n] = r_dev[3*shared_idx + n];
        }

        for (unsigned int n = 0; n < 3; ++n) {
            sj[tx][n] = s_dev[3*shared_idx + n];
        }
      }
      // sync to make sure the shared data is filled from all block threads
      __syncthreads();

      if (idx < num_spins) {

        for (unsigned int t = 0; t < block_size; ++t) {
          unsigned int jdx = t + block_size * b;

          if (jdx < num_spins && jdx != idx) {

            float r_ij[3];

            displacement(ri, rj[t], r_ij);

            const float r_abs = sqrtf(r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]);

            if (r_abs > dev_r_cutoff) continue;

            float sj_dot_rhat = (sj[t][0] * r_ij[0] + sj[t][1] * r_ij[1] + sj[t][2] * r_ij[2]);

            for (int n = 0; n < 3; ++n) {
                h[n] += pre * mus_pre[t] * (3.0f * r_ij[n] * sj_dot_rhat  - r_abs * r_abs * sj[t][n]) / (r_abs * r_abs * r_abs * r_abs * r_abs);
            }
          }
        }
      }
    } 

  if (idx < num_spins) {
    // write to global memory
    #pragma unroll
    for (unsigned int n = 0; n < 3; ++n) {
        h_dev[3*idx + n] = h[n];
    }
  }
}


#endif  // JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H
