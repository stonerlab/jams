#ifndef JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H
#define JAMS_HAMILTONIAN_DIPOLE_BRUTEFORCE_KERNEL_H

#include "jams/core/cuda_defs.h"
#include "jams/core/cuda_vector_ops.h"

__constant__ float dev_super_unit_cell[3][3];
__constant__ float dev_super_unit_cell_inv[3][3];
__constant__ bool   dev_super_cell_pbc[3];
__constant__ float dev_dipole_prefactor;
__constant__ float dev_r_cutoff;

constexpr unsigned int block_size = 128;

//-----------------------------------------------------------------------------
// Apply the minimum image convetion on a displacement vector in the fractional 
// coordinate space
//
// Note: displacement vector must be in fractional coordinates
//
// ref: W. Smith, CCP5 Information Quarterly for Computer Simulation of 
//      Condensed Phases (1989).
//-----------------------------------------------------------------------------
__device__ 
inline void ApplyMinimumImageConvention(const bool pbc[3], 
    float displacement[3]) {
  #pragma unroll
  for (unsigned int n = 0; n < 3; ++n) {
    if (pbc[n]) { 
      displacement[n] = 
          displacement[n] - trunc(2.0 * displacement[n]);
    }
  }
}

//-----------------------------------------------------------------------------
// Calculate the diplacement vector between position r_i and r_j in cartesian
// space accounting for any periodic boundary conditions
//-----------------------------------------------------------------------------
__device__ 
inline void CalculateDisplacementVector(const float r_i_cartesian[3],
    const float rj_cartesian[3], float dr_cartesian[3]) {
  float dr_fractional[3];

  #pragma unroll
  for (unsigned int n = 0; n < 3; ++n) {
    dr_cartesian[n] = r_i_cartesian[n] - rj_cartesian[n];
  }
  // transform cartesian vector into fractional space
  matmul(dev_super_unit_cell_inv, dr_cartesian, dr_fractional);

  ApplyMinimumImageConvention(dev_super_cell_pbc, dr_fractional);

  // transform fractional vector back to cartesian space
  matmul(dev_super_unit_cell, dr_fractional, dr_cartesian);
}

//-----------------------------------------------------------------------------
// Calculate the field contribution from spin j with displacement vector r_ij
//-----------------------------------------------------------------------------
__device__ 
inline void CalculateFieldElement(const float s_j[3], const float r_ij[3], 
    const float prefactor, const float r_abs_sq, float h[3]) {
  const float d_r_abs = rsqrtf(r_abs_sq);
  const float s_j_dot_rhat = dot(s_j, r_ij);
  const float w0 = prefactor * d_r_abs * d_r_abs * d_r_abs * d_r_abs * d_r_abs;

  #pragma unroll
  for (unsigned int n = 0; n < 3; ++n) {
    h[n] = w0 * (3.0f * r_ij[n] * s_j_dot_rhat  - r_abs_sq * s_j[n]);
  }
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

          CalculateDisplacementVector(ri, rj, r_ij);

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

//-----------------------------------------------------------------------------
// Calculate dipole fields for all spins using a bruteforce algorithm
//
// Notes: This implementation uses a shared memory cache so that each thread
//        block can access a common pool of spins and positions. The calculation
//        loop is then unrolled 4x by hand giving a big perfomance boost because
//        we avoid warp divergence in the if statements until later in the
//        calculation and also give the processors lots of maths to do while
//        waiting for memory loads. 
//
//        This kernel is using mixed precision to avoid externally converting 
//        the spin array into floats, but using floats internally for faster
//        maths.
//-----------------------------------------------------------------------------
__global__ 
void DipoleBruteforceKernel(
    const double * __restrict__ s_dev,
    const float * __restrict__ r_dev,
    const float * __restrict__ mus_dev,
    const unsigned int num_spins,
          double * __restrict__ h_dev)
{
    const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int thread_idx = threadIdx.x;

    const float r_cutoff_sq = dev_r_cutoff * dev_r_cutoff; 

    __shared__ float r_j[block_size][3];
    __shared__ float s_j[block_size][3];
    __shared__ float mus_pre[block_size];

    float prefactor, r_i[3];

    if (i < num_spins) {
      // constant is kB * mu_0 / 4 pi
      prefactor = mus_dev[i] * dev_dipole_prefactor;

      for (unsigned int n = 0; n < 3; ++n) {
          r_i[n] = r_dev[3*i + n];
      }
    }

    const unsigned int num_blocks = (num_spins + block_size - 1) / block_size;

    double h[3] = {0.0, 0.0, 0.0};
    for (unsigned int block_idx = 0; block_idx < num_blocks; ++block_idx) {
      //------------------------------------------------------------------------
      // for every block load up the shared memory with data
      //------------------------------------------------------------------------
      const unsigned int shared_idx = thread_idx + block_size * block_idx;
      // sync to make sure we aren't using a dirty cache when the warp diverges
      __syncthreads();
      // only load data which exists
      if (shared_idx < num_spins) {
        mus_pre[thread_idx] = prefactor * mus_dev[shared_idx];
        #pragma unroll
        for (unsigned int n = 0; n < 3; ++n) {
            r_j[thread_idx][n] = r_dev[3*shared_idx + n];
        }
        #pragma unroll
        for (unsigned int n = 0; n < 3; ++n) {
            s_j[thread_idx][n] = s_dev[3*shared_idx + n];
        }
      }
      // sync to make sure the shared data is filled from all block threads
      __syncthreads();
      //------------------------------------------------------------------------


      if (i < num_spins) {
          // do lots of maths on the pool of spins and positions to hide memory 
          // transfer latency
          for (unsigned int t = 0; t < block_size/4; ++t) {

            // shared memory indices
            const int t1 = 4*t;
            const int t2 = 4*t+1;
            const int t3 = 4*t+2;
            const int t4 = 4*t+3;

            // spin indices
            const int j1 = (block_size * block_idx + t1);
            const int j2 = (block_size * block_idx + t2);
            const int j3 = (block_size * block_idx + t3);
            const int j4 = (block_size * block_idx + t4);

            const float pre1 = mus_pre[t1];
            const float pre2 = mus_pre[t2];
            const float pre3 = mus_pre[t3];
            const float pre4 = mus_pre[t4];

            float r_ij1[3], r_ij2[3], r_ij3[3], r_ij4[3];

            CalculateDisplacementVector(r_i, r_j[t1], r_ij1);
            const float r_abs1 = abs(r_ij1);
            CalculateDisplacementVector(r_i, r_j[t2], r_ij2);
            const float r_abs2 = abs(r_ij2);
            CalculateDisplacementVector(r_i, r_j[t3], r_ij3);
            const float r_abs3 = abs(r_ij3);
            CalculateDisplacementVector(r_i, r_j[t4], r_ij4);
            const float r_abs4 = abs(r_ij4);

            float h1[3] = {0.0, 0.0, 0.0};
            float h2[3] = {0.0, 0.0, 0.0};
            float h3[3] = {0.0, 0.0, 0.0};
            float h4[3] = {0.0, 0.0, 0.0};

            if (r_abs1 <= r_cutoff_sq && i != j1 && j1 < num_spins) {
              CalculateFieldElement(s_j[t1], r_ij1, pre1, r_abs1, h1);
            }
            if (r_abs2 <= r_cutoff_sq && i != j2 && j2 < num_spins) {
              CalculateFieldElement(s_j[t2], r_ij2, pre2, r_abs2, h2);
            }
            if (r_abs3 <= r_cutoff_sq && i != j3 && j3 < num_spins) {
              CalculateFieldElement(s_j[t3], r_ij3, pre3, r_abs3, h3);
            }
            if (r_abs4 <= r_cutoff_sq && i != j4 && j4 < num_spins) {
              CalculateFieldElement(s_j[t4], r_ij4, pre4, r_abs4, h4);
            }

            #pragma unroll
            for (unsigned int n = 0; n < 3; ++n) {
              h[n] = h[n] + h1[n] + h2[n] + h3[n] + h4[n];
            }
          }
        }
      } // end block loop

  if (i < num_spins) {
    // write to global memory
    #pragma unroll
    for (unsigned int n = 0; n < 3; ++n) {
        h_dev[3*i + n] = h[n];
    }
  }
}


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

            CalculateDisplacementVector(ri, rj[t], r_ij);

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
