#include <cuda_runtime.h>
#include <jams/common.h>
#include <jams/core/types.h>
#include "jams/cuda/cuda_device_vector_ops.h"
#include <jams/cuda/cuda_array_kernels.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/containers/multiarray.h>
#include <jams/containers/sparse_matrix.h>


__global__ void undamped_llg_spin_derivative_kernel
        (const int num_spins,
         const double *spins,
         const jams::Real *field,
         const jams::Real *gyro,
         const jams::Real *mus,
         double *spin_derivative
        ) {

  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < num_spins) {
    if (mus[i] == jams::Real(0.0)) {
      for (int n = 0; n < 3; ++n) {
        spin_derivative[3*i + n] = 0.0;
      }
      return;
    }

    const double s_i[3] = {spins[3*i + 0], spins[3*i + 1], spins[3*i + 2]};
    const double h_i[3] = {
        static_cast<double>(field[3*i + 0]) / static_cast<double>(mus[i]),
        static_cast<double>(field[3*i + 1]) / static_cast<double>(mus[i]),
        static_cast<double>(field[3*i + 2]) / static_cast<double>(mus[i])
    };

    double sxh[3];
    cross_product(s_i, h_i, sxh);

    for (int n = 0; n < 3; ++n) {
      spin_derivative[3*i + n] = -static_cast<double>(gyro[i]) * sxh[n];
    }
  }
}

jams::Vec<double, 3> execute_cuda_thermal_current_kernel(
    CudaStream &stream,
    const jams::MultiArray<double, 2>& spins,
    const jams::MultiArray<jams::Real, 2>& field,
    const jams::MultiArray<jams::Real, 1>& gyro,
    const jams::MultiArray<jams::Real, 1>& mus,
    jams::SparseMatrix<double>& energy_current_operator_rx,
    jams::SparseMatrix<double>& energy_current_operator_ry,
    jams::SparseMatrix<double>& energy_current_operator_rz,
    const double volume,
    jams::MultiArray<double, 2>& dev_spin_derivative,
    jams::MultiArray<double, 2>& dev_energy_current_rx,
    jams::MultiArray<double, 2>& dev_energy_current_ry,
    jams::MultiArray<double, 2>& dev_energy_current_rz,
    jams::MultiArray<double, 1>& dev_energy_current_dot) {

  dim3 block_size;
  block_size.x = 64;
  dim3 grid_size;
  grid_size.x = (spins.extent(0) + block_size.x - 1) / block_size.x;

  undamped_llg_spin_derivative_kernel<<<grid_size, block_size, 0, stream.get()>>>(
      spins.extent(0),
      spins.device_data(),
      field.device_data(),
      gyro.device_data(),
      mus.device_data(),
      dev_spin_derivative.mutable_device_data());
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  energy_current_operator_rx.multiply_gpu(
      spins, dev_energy_current_rx, jams::instance().cusparse_handle(), stream.get());
  energy_current_operator_ry.multiply_gpu(
      spins, dev_energy_current_ry, jams::instance().cusparse_handle(), stream.get());
  energy_current_operator_rz.multiply_gpu(
      spins, dev_energy_current_rz, jams::instance().cusparse_handle(), stream.get());

  const double prefactor = -0.5 / volume;

  cuda_array_dot_product(
      spins.extent(0),
      prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_rx.device_data(),
      dev_energy_current_dot.mutable_device_data(),
      stream.get());
  double j_rx = cuda_reduce_array(dev_energy_current_dot.device_data(), spins.extent(0), stream.get());

  cuda_array_dot_product(
      spins.extent(0),
      prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_ry.device_data(),
      dev_energy_current_dot.mutable_device_data(),
      stream.get());
  double j_ry = cuda_reduce_array(dev_energy_current_dot.device_data(), spins.extent(0), stream.get());

  cuda_array_dot_product(
      spins.extent(0),
      prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_rz.device_data(),
      dev_energy_current_dot.mutable_device_data(),
      stream.get());
  double j_rz = cuda_reduce_array(dev_energy_current_dot.device_data(), spins.extent(0), stream.get());

  return {j_rx, j_ry, j_rz};
}
