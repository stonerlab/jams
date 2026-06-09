#include <jams/containers/block_sparse_interaction_matrix.h>
#include <jams/cuda/cuda_common.h>
#include <jams/helpers/mixed_precision.h>

#if HAS_CUDA

namespace {

template <typename SpinT, typename ValueT, int Storage>
__global__ void block_sparse_interaction_multiply_kernel(
    int num_spins,
    const int32_t* rows,
    const int32_t* cols,
    const ValueT* values,
    const SpinT* spins,
    ValueT* field) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= num_spins) {
    return;
  }

  ValueT hx = 0;
  ValueT hy = 0;
  ValueT hz = 0;

  constexpr int components =
      Storage == static_cast<int>(jams::InteractionTensorStorage::Isotropic) ? 1 :
      Storage == static_cast<int>(jams::InteractionTensorStorage::Anisotropic) ? 3 :
      Storage == static_cast<int>(jams::InteractionTensorStorage::Symmetric) ? 6 :
      Storage == static_cast<int>(jams::InteractionTensorStorage::Antisymmetric) ? 3 : 9;

  for (auto n = rows[i]; n < rows[i + 1]; ++n) {
    const auto j = cols[n];
    const auto spin_base = 3 * j;
    const ValueT sx = static_cast<ValueT>(spins[spin_base + 0]);
    const ValueT sy = static_cast<ValueT>(spins[spin_base + 1]);
    const ValueT sz = static_cast<ValueT>(spins[spin_base + 2]);
    const ValueT* v = values + n * components;

    if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Isotropic)) {
      hx += v[0] * sx;
      hy += v[0] * sy;
      hz += v[0] * sz;
    } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Anisotropic)) {
      hx += v[0] * sx;
      hy += v[1] * sy;
      hz += v[2] * sz;
    } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Symmetric)) {
      hx += v[0] * sx + v[1] * sy + v[2] * sz;
      hy += v[1] * sx + v[3] * sy + v[4] * sz;
      hz += v[2] * sx + v[4] * sy + v[5] * sz;
    } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Antisymmetric)) {
      hx += v[0] * sy + v[1] * sz;
      hy += -v[0] * sx + v[2] * sz;
      hz += -v[1] * sx - v[2] * sy;
    } else {
      hx += v[0] * sx + v[1] * sy + v[2] * sz;
      hy += v[3] * sx + v[4] * sy + v[5] * sz;
      hz += v[6] * sx + v[7] * sy + v[8] * sz;
    }
  }

  const auto field_base = 3 * i;
  field[field_base + 0] = hx;
  field[field_base + 1] = hy;
  field[field_base + 2] = hz;
}

template <typename SpinT, typename ValueT>
void launch_block_sparse_interaction_multiply(
    const jams::BlockSparseInteractionMatrix<ValueT>& matrix,
    const jams::MultiArray<SpinT, 2>& spins,
    jams::MultiArray<ValueT, 2>& field,
    cudaStream_t stream) {
  const dim3 block_size = {128, 1, 1};
  const dim3 grid_size = cuda_grid_size(block_size, {static_cast<unsigned int>(matrix.num_rows()), 1, 1});

  switch (matrix.storage()) {
    case jams::InteractionTensorStorage::Isotropic:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::Isotropic)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::Anisotropic:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::Anisotropic)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::Symmetric:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::Symmetric)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::Antisymmetric:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::Antisymmetric)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::General:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::General)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::Auto:
      throw std::runtime_error("cannot multiply auto tensor storage on GPU");
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

}  // namespace

namespace jams {

template <>
void BlockSparseInteractionMatrix<jams::Real>::multiply_gpu(
    const MultiArray<double, 2>& spins,
    MultiArray<jams::Real, 2>& field,
    cudaStream_t stream) const {
  launch_block_sparse_interaction_multiply(*this, spins, field, stream);
}

template <>
void BlockSparseInteractionMatrix<jams::Real>::multiply_gpu(
    const MultiArray<float, 2>& spins,
    MultiArray<jams::Real, 2>& field,
    cudaStream_t stream) const {
  launch_block_sparse_interaction_multiply(*this, spins, field, stream);
}

}  // namespace jams

#endif  // HAS_CUDA
