#include <cuda.h>
#include <jams/core/types.h>
#include "jams/cuda/cuda_device_vector_ops.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <jams/cuda/cuda_array_kernels.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/cuda/cuda_array_kernels.h>
#include <jams/containers/multiarray.h>
#include <jams/containers/interaction_matrix.h>


__global__ void thermal_current_kernel
        (const int num_spins,
         const double *spins,
         const int *index_pointers,
         const int *index_data,
         const double *value_data,
         double *thermal_current_rx,
         double *thermal_current_ry,
         double *thermal_current_rz
        ) {

  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < num_spins) {
    double jq[3] = {0.0, 0.0, 0.0};

    const int begin = index_pointers[i];
    const int end = index_pointers[i + 1];

    const double s_i[3] = {spins[3*i + 0], spins[3*i + 1], spins[3*i + 2]};

    for (int n = begin; n < end; ++n) {
      const int j = index_data[3*n];
      const int k = index_data[3*n + 1];
      const int val_key = index_data[3*n + 2];
      assert(j < num_spins);
      assert(k < num_spins);
//      assert(i != j && j != k && i != k);


      const double s_j[3] = {spins[3*j + 0], spins[3*j + 1], spins[3*j + 2]};
      const double s_k[3] = {spins[3*k + 0], spins[3*k + 1], spins[3*k + 2]};

      for (int m = 0; m < 3; ++m) {
        jq[m] += value_data[3*val_key + m] * scalar_triple_product(s_i, s_j, s_k);
      }
    }

    thermal_current_rx[i] = jq[0];
    thermal_current_ry[i] = jq[1];
    thermal_current_rz[i] = jq[2];
  }
}

Vec3 execute_cuda_thermal_current_kernel(
    CudaStream &stream,
    const jams::MultiArray<double, 2>& spins,
    const jams::InteractionMatrix<Vec3, double>& interaction_matrix,
    jams::MultiArray<double, 1>& dev_thermal_current_rx,
    jams::MultiArray<double, 1>& dev_thermal_current_ry,
    jams::MultiArray<double, 1>& dev_thermal_current_rz) {

  dim3 block_size;
  block_size.x = 64;
  dim3 grid_size;
  grid_size.x = (spins.size(0) + block_size.x - 1) / block_size.x;

  thermal_current_kernel<<<grid_size, block_size, 0, stream.get()>>>(
      spins.size(0),
      spins.device_data(),
      interaction_matrix.row_device_data(),
      interaction_matrix.index_device_data(),
      interaction_matrix.val_device_data(),
      dev_thermal_current_rx.device_data(),
      dev_thermal_current_ry.device_data(),
      dev_thermal_current_rz.device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  // triple counting in the sum
  double j_rx = 0.5 * cuda_reduce_array(dev_thermal_current_rx.device_data(), spins.size(0));
  double j_ry = 0.5 * cuda_reduce_array(dev_thermal_current_ry.device_data(), spins.size(0));
  double j_rz = 0.5 * cuda_reduce_array(dev_thermal_current_rz.device_data(), spins.size(0));

  return {j_rx, j_ry, j_rz};
}

