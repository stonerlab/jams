#include <cuda_runtime.h>
#include <jams/cuda/cuda_array_kernels.h>

#include "jams/core/types.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/cuda/cuda_stream.h"

__global__ void spin_current_kernel
        (const int num_spins,
         const double *spins,
         const jams::Real *gyro,
         const jams::Real *mus,
         const double *Jrij,
         const int *col_pointers,
         const int *col_indicies,
         double *spin_current_rx_z,
         double *spin_current_ry_z,
         double *spin_current_rz_z
        ) {

  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < num_spins) {
    const double s_i[3] = {spins[3*i + 0], spins[3*i + 1], spins[3*i + 2]};

    double js_z[3] = {0.0, 0.0, 0.0};

    const int begin = col_pointers[i];
    const int end = col_pointers[i + 1];

    for (auto n = begin; n < end; ++n) {
      const auto j = col_indicies[n];

      const double s_j[3] = {spins[3*j + 0], spins[3*j + 1], spins[3*j + 2]};

      const double Jij_rij[3] = {Jrij[3*n + 0], Jrij[3*n + 1], Jrij[3*n + 2]};

      double s_i_cross_s_j_z = cross_product_z(s_i, s_j);

      for (auto m = 0; m < 3; ++m) {
        js_z[m] += Jij_rij[m] * s_i_cross_s_j_z;
      }
    }

    const double prefactor = gyro[i] / mus[i];

    spin_current_rx_z[i] = prefactor*js_z[0];
    spin_current_ry_z[i] = prefactor*js_z[1];
    spin_current_rz_z[i] = prefactor*js_z[2];
  }
}

Vec3 execute_cuda_spin_current_kernel(
        CudaStream &stream,
        const int num_spins,
        const double *dev_spins,
        const jams::Real *dev_gyro,
        const jams::Real *dev_mus,
        const double *dev_Jrij,
        const int *dev_col_pointers,
        const int *dev_col_indicies,
        double *dev_spin_current_rx_z,
        double *dev_spin_current_ry_z,
        double *dev_spin_current_rz_z
                                     ) {
  dim3 block_size;
  block_size.x = 64;
  dim3 grid_size;
  grid_size.x = (num_spins + block_size.x - 1) / block_size.x;

  spin_current_kernel<<<grid_size, block_size, 0, stream.get()>>>(
          num_spins,
          dev_spins,
          dev_gyro,
          dev_mus,
          dev_Jrij,
          dev_col_pointers,
          dev_col_indicies,
          dev_spin_current_rx_z,
          dev_spin_current_ry_z,
          dev_spin_current_rz_z);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  double j_rx_z = 0.5*cuda_reduce_array(dev_spin_current_rx_z, num_spins);
  double j_ry_z = 0.5*cuda_reduce_array(dev_spin_current_ry_z, num_spins);
  double j_rz_z = 0.5*cuda_reduce_array(dev_spin_current_rz_z, num_spins);
  
  return {j_rx_z, j_ry_z, j_rz_z};
}

