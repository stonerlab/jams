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

template <typename FieldType>
__global__ void thermal_current_dot_product_kernel(
    const int num_spins,
    const double current_density_prefactor,
    const double* spin_derivative,
    const FieldType* energy_current,
    double* out) {
  const int i = blockIdx.x*blockDim.x+threadIdx.x;

  if (i < num_spins) {
    const int base = 3 * i;
    double dot = 0.0;
    for (int n = 0; n < 3; ++n) {
      dot += spin_derivative[base + n] * static_cast<double>(energy_current[base + n]);
    }
    out[i] = current_density_prefactor * dot;
  }
}

template <typename FieldType>
double reduce_thermal_current_field(
    CudaStream& stream,
    const int num_spins,
    const double current_density_prefactor,
    const double* spin_derivative,
    const FieldType* energy_current,
    double* dot_buffer) {
  dim3 block_size;
  block_size.x = 128;
  dim3 grid_size;
  grid_size.x = (num_spins + block_size.x - 1) / block_size.x;

  thermal_current_dot_product_kernel<<<grid_size, block_size, 0, stream.get()>>>(
      num_spins,
      current_density_prefactor,
      spin_derivative,
      energy_current,
      dot_buffer);
  DEBUG_CHECK_CUDA_ASYNC_STATUS;

  return cuda_reduce_array(dot_buffer, num_spins, stream.get());
}

jams::Vec<double, 3> execute_cuda_thermal_current_field_reduction(
    CudaStream& stream,
    const double current_density_prefactor,
    const jams::MultiArray<double, 2>& dev_spin_derivative,
    const jams::MultiArray<jams::Real, 2>& dev_energy_current_rx,
    const jams::MultiArray<jams::Real, 2>& dev_energy_current_ry,
    const jams::MultiArray<jams::Real, 2>& dev_energy_current_rz,
    jams::MultiArray<double, 1>& dev_energy_current_dot) {
  const int num_spins = dev_spin_derivative.extent(0);
  const double j_rx = reduce_thermal_current_field(
      stream,
      num_spins,
      current_density_prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_rx.device_data(),
      dev_energy_current_dot.mutable_device_data());
  const double j_ry = reduce_thermal_current_field(
      stream,
      num_spins,
      current_density_prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_ry.device_data(),
      dev_energy_current_dot.mutable_device_data());
  const double j_rz = reduce_thermal_current_field(
      stream,
      num_spins,
      current_density_prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_rz.device_data(),
      dev_energy_current_dot.mutable_device_data());

  return {j_rx, j_ry, j_rz};
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
    const double current_density_prefactor,
    const bool has_sparse_energy_current_operator,
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

  if (!has_sparse_energy_current_operator) {
    return {0.0, 0.0, 0.0};
  }

  energy_current_operator_rx.multiply_gpu(
      spins, dev_energy_current_rx, jams::instance().cusparse_handle(), stream.get());
  energy_current_operator_ry.multiply_gpu(
      spins, dev_energy_current_ry, jams::instance().cusparse_handle(), stream.get());
  energy_current_operator_rz.multiply_gpu(
      spins, dev_energy_current_rz, jams::instance().cusparse_handle(), stream.get());

  const double j_rx = reduce_thermal_current_field(
      stream,
      spins.extent(0),
      current_density_prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_rx.device_data(),
      dev_energy_current_dot.mutable_device_data());

  const double j_ry = reduce_thermal_current_field(
      stream,
      spins.extent(0),
      current_density_prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_ry.device_data(),
      dev_energy_current_dot.mutable_device_data());

  const double j_rz = reduce_thermal_current_field(
      stream,
      spins.extent(0),
      current_density_prefactor,
      dev_spin_derivative.device_data(),
      dev_energy_current_rz.device_data(),
      dev_energy_current_dot.mutable_device_data());

  return {j_rx, j_ry, j_rz};
}
