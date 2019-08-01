#include <cuda.h>
#include <jams/core/types.h>
#include "jams/cuda/cuda_device_vector_ops.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <jams/cuda/cuda_array_kernels.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/cuda/cuda_array_kernels.h>


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
      const int j = index_data[2*n];
      const int k = index_data[2*n + 1];
      assert(j < num_spins);
      assert(k < num_spins);

      const double s_j[3] = {spins[3*j + 0], spins[3*j + 1], spins[3*j + 2]};
      const double s_k[3] = {spins[3*k + 0], spins[3*k + 1], spins[3*k + 2]};

      for (int m = 0; m < 3; ++m) {
        jq[m] += value_data[3*n + m] * scalar_triple_product(s_i, s_j, s_k);
      }
    }

    thermal_current_rx[i] = jq[0];
    thermal_current_ry[i] = jq[1];
    thermal_current_rz[i] = jq[2];
  }
}

Vec3 execute_cuda_thermal_current_kernel(
        CudaStream &stream,
        const int num_spins,
        const double *dev_spins,
        const int *index_pointers,
        const int *index_data,
        const double *value_data,
        double *dev_thermal_current_rx,
        double *dev_thermal_current_ry,
        double *dev_thermal_current_rz
                                        ) {

  assert(dev_spins != nullptr);
  assert(index_pointers != nullptr);
  assert(index_data != nullptr);
  assert(value_data != nullptr);
  assert(dev_thermal_current_rx != nullptr);
  assert(dev_thermal_current_ry != nullptr);
  assert(dev_thermal_current_rz != nullptr);

  dim3 block_size;
  block_size.x = 64;
  dim3 grid_size;
  grid_size.x = (num_spins + block_size.x - 1) / block_size.x;

  thermal_current_kernel<<<grid_size, block_size, 0, stream.get()>>>(
          num_spins,
                  dev_spins,
                  index_pointers,
                  index_data,
                  value_data,
                  dev_thermal_current_rx,
                  dev_thermal_current_ry,
                  dev_thermal_current_rz);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  // triple counting in the sum
  double j_rx = 0.5 * cuda_reduce_array(dev_thermal_current_rx, num_spins);
  double j_ry = 0.5 * cuda_reduce_array(dev_thermal_current_ry, num_spins);
  double j_rz = 0.5 * cuda_reduce_array(dev_thermal_current_rz, num_spins);

  return {j_rx, j_ry, j_rz};
}

