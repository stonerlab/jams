#include <cuda.h>
#include <jams/cuda/cuda_array_kernels.h>

#include "jams/core/types.h"
#include "jams/cuda/cuda_device_vector_ops.h"
#include "jams/cuda/cuda_stream.h"

__global__ void spin_current_kernel
        (const int num_spins,
         const double *spins,
         const double *Jrij,
         const int *col_pointers,
         const int *col_indicies,
         double *spin_current_rx_x,
         double *spin_current_rx_y,
         double *spin_current_rx_z,
         double *spin_current_ry_x,
         double *spin_current_ry_y,
         double *spin_current_ry_z,
         double *spin_current_rz_x,
         double *spin_current_rz_y,
         double *spin_current_rz_z
        ) {

  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < num_spins) {
    const double s_i[3] = {spins[3*i + 0], spins[3*i + 1], spins[3*i + 2]};

    double js_rx[3] = {0.0, 0.0, 0.0};
    double js_ry[3] = {0.0, 0.0, 0.0};
    double js_rz[3] = {0.0, 0.0, 0.0};

    const int begin = col_pointers[i];
    const int end = col_pointers[i + 1];

    for (auto n = begin; n < end; ++n) {
      const auto j = col_indicies[n];

      const double s_j[3] = {spins[3*j + 0], spins[3*j + 1], spins[3*j + 2]};

      const double Jij_rij[3] = {Jrij[3*n + 0], Jrij[3*n + 1], Jrij[3*n + 2]};

      double s_i_cross_s_j[3];
      cross_product(s_i, s_j, s_i_cross_s_j);

      for (auto m = 0; m < 3; ++m) {
        js_rx[m] += Jij_rij[0] * s_i_cross_s_j[m];
        js_ry[m] += Jij_rij[1] * s_i_cross_s_j[m];
        js_rz[m] += Jij_rij[2] * s_i_cross_s_j[m];
      }
    }

    spin_current_rx_x[i] = js_rx[0];
    spin_current_rx_y[i] = js_rx[1];
    spin_current_rx_z[i] = js_rx[2];

    spin_current_ry_x[i] = js_ry[0];
    spin_current_ry_y[i] = js_ry[1];
    spin_current_ry_z[i] = js_ry[2];

    spin_current_rz_x[i] = js_rz[0];
    spin_current_rz_y[i] = js_rz[1];
    spin_current_rz_z[i] = js_rz[2];
  }
}

Mat3 execute_cuda_spin_current_kernel(
        CudaStream &stream,
        const int num_spins,
        const double *dev_spins,
        const double *dev_Jrij,
        const int *dev_col_pointers,
        const int *dev_col_indicies,
        double *dev_spin_current_rx_x,
        double *dev_spin_current_rx_y,
        double *dev_spin_current_rx_z,
        double *dev_spin_current_ry_x,
        double *dev_spin_current_ry_y,
        double *dev_spin_current_ry_z,
        double *dev_spin_current_rz_x,
        double *dev_spin_current_rz_y,
        double *dev_spin_current_rz_z
                                     ) {
  dim3 block_size;
  block_size.x = 64;
  dim3 grid_size;
  grid_size.x = (num_spins + block_size.x - 1) / block_size.x;

  spin_current_kernel<<<grid_size, block_size, 0, stream.get()>>>(
          num_spins,
                  dev_spins,
                  dev_Jrij,
                  dev_col_pointers,
                  dev_col_indicies,
                  dev_spin_current_rx_x,
                  dev_spin_current_rx_y,
                  dev_spin_current_rx_z,
                  dev_spin_current_ry_x,
                  dev_spin_current_ry_y,
                  dev_spin_current_ry_z,
                  dev_spin_current_rz_x,
                  dev_spin_current_rz_y,
                  dev_spin_current_rz_z);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  // double counting in the sum
  double j_rx_x = 0.5*cuda_reduce_array(dev_spin_current_rx_x, num_spins);
  double j_rx_y = 0.5*cuda_reduce_array(dev_spin_current_rx_y, num_spins);
  double j_rx_z = 0.5*cuda_reduce_array(dev_spin_current_rx_z, num_spins);

  double j_ry_x = 0.5*cuda_reduce_array(dev_spin_current_ry_x, num_spins);
  double j_ry_y = 0.5*cuda_reduce_array(dev_spin_current_ry_y, num_spins);
  double j_ry_z = 0.5*cuda_reduce_array(dev_spin_current_ry_z, num_spins);

  double j_rz_x = 0.5*cuda_reduce_array(dev_spin_current_rz_x, num_spins);
  double j_rz_y = 0.5*cuda_reduce_array(dev_spin_current_rz_y, num_spins);
  double j_rz_z = 0.5*cuda_reduce_array(dev_spin_current_rz_z, num_spins);
  
  return {j_rx_x, j_rx_y, j_rx_z,
          j_ry_x, j_ry_y, j_ry_z,
          j_rz_x, j_rz_y, j_rz_z};
}

