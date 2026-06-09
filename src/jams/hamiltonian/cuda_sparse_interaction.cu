#include <jams/containers/block_sparse_interaction_matrix.h>
#include <jams/cuda/cuda_common.h>
#include <jams/helpers/mixed_precision.h>

#if HAS_CUDA

namespace {

template <typename SpinT, typename ValueT, int Storage>
__global__ void block_sparse_interaction_multiply_kernel(
    int num_spins,
    int num_blocks,
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

  for (auto n = rows[i]; n < rows[i + 1]; ++n) {
    const auto j = cols[n];
    const auto spin_base = 3 * j;
    const ValueT sx = static_cast<ValueT>(spins[spin_base + 0]);
    const ValueT sy = static_cast<ValueT>(spins[spin_base + 1]);
    const ValueT sz = static_cast<ValueT>(spins[spin_base + 2]);

    if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Isotropic)) {
      const ValueT j0 = values[n];
      hx += j0 * sx;
      hy += j0 * sy;
      hz += j0 * sz;
    } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Anisotropic)) {
      hx += values[n] * sx;
      hy += values[num_blocks + n] * sy;
      hz += values[2 * num_blocks + n] * sz;
    } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Symmetric)) {
      hx += values[n] * sx + values[num_blocks + n] * sy + values[2 * num_blocks + n] * sz;
      hy += values[num_blocks + n] * sx + values[3 * num_blocks + n] * sy + values[4 * num_blocks + n] * sz;
      hz += values[2 * num_blocks + n] * sx + values[4 * num_blocks + n] * sy + values[5 * num_blocks + n] * sz;
    } else if constexpr (Storage == static_cast<int>(jams::InteractionTensorStorage::Antisymmetric)) {
      hx += values[n] * sy + values[num_blocks + n] * sz;
      hy += -values[n] * sx + values[2 * num_blocks + n] * sz;
      hz += -values[num_blocks + n] * sx - values[2 * num_blocks + n] * sy;
    } else {
      hx += values[n] * sx + values[num_blocks + n] * sy + values[2 * num_blocks + n] * sz;
      hy += values[3 * num_blocks + n] * sx + values[4 * num_blocks + n] * sy + values[5 * num_blocks + n] * sz;
      hz += values[6 * num_blocks + n] * sx + values[7 * num_blocks + n] * sy + values[8 * num_blocks + n] * sz;
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
              matrix.num_rows(), matrix.num_blocks(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::Anisotropic:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::Anisotropic)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.num_blocks(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::Symmetric:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::Symmetric)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.num_blocks(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::Antisymmetric:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::Antisymmetric)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.num_blocks(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
              spins.device_data(), field.mutable_device_data());
      break;
    case jams::InteractionTensorStorage::General:
      block_sparse_interaction_multiply_kernel<SpinT, ValueT, static_cast<int>(jams::InteractionTensorStorage::General)>
          <<<grid_size, block_size, 0, stream>>>(
              matrix.num_rows(), matrix.num_blocks(), matrix.row_device_data(), matrix.col_device_data(), matrix.val_device_data(),
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
