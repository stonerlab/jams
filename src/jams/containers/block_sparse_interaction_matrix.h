#ifndef JAMS_CONTAINERS_BLOCK_SPARSE_INTERACTION_MATRIX_H
#define JAMS_CONTAINERS_BLOCK_SPARSE_INTERACTION_MATRIX_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <jams/containers/mat3.h>
#include <jams/containers/multiarray.h>
#include <jams/helpers/utils.h>

namespace jams {

enum class InteractionTensorStorage {
  Auto,
  Isotropic,
  Anisotropic,
  Symmetric,
  Antisymmetric,
  General
};

inline std::string to_string(const InteractionTensorStorage storage) {
  switch (storage) {
    case InteractionTensorStorage::Auto:
      return "auto";
    case InteractionTensorStorage::Isotropic:
      return "isotropic";
    case InteractionTensorStorage::Anisotropic:
      return "anisotropic";
    case InteractionTensorStorage::Symmetric:
      return "symmetric";
    case InteractionTensorStorage::Antisymmetric:
      return "antisymmetric";
    case InteractionTensorStorage::General:
      return "general";
  }
  throw std::invalid_argument("unknown interaction tensor storage");
}

inline InteractionTensorStorage interaction_tensor_storage_from_string(std::string value) {
  value = lowercase(value);
  if (value == "auto") {
    return InteractionTensorStorage::Auto;
  }
  if (value == "isotropic" || value == "scalar") {
    return InteractionTensorStorage::Isotropic;
  }
  if (value == "anisotropic" || value == "diagonal") {
    return InteractionTensorStorage::Anisotropic;
  }
  if (value == "symmetric") {
    return InteractionTensorStorage::Symmetric;
  }
  if (value == "antisymmetric" || value == "skew" || value == "skew-symmetric") {
    return InteractionTensorStorage::Antisymmetric;
  }
  if (value == "general" || value == "tensor") {
    return InteractionTensorStorage::General;
  }
  throw std::runtime_error("unknown tensor_storage: " + value);
}

inline int interaction_tensor_storage_components(const InteractionTensorStorage storage) {
  switch (storage) {
    case InteractionTensorStorage::Auto:
      throw std::invalid_argument("auto tensor storage has no component count");
    case InteractionTensorStorage::Isotropic:
      return 1;
    case InteractionTensorStorage::Anisotropic:
      return 3;
    case InteractionTensorStorage::Symmetric:
      return 6;
    case InteractionTensorStorage::Antisymmetric:
      return 3;
    case InteractionTensorStorage::General:
      return 9;
  }
  throw std::invalid_argument("unknown interaction tensor storage");
}

template<typename T>
class BlockSparseInteractionMatrix {
public:
  class Builder;

  using value_type = T;
  using index_type = int32_t;
  using index_container = MultiArray<index_type, 1>;
  using value_container = MultiArray<value_type, 1>;

  BlockSparseInteractionMatrix() = default;

  BlockSparseInteractionMatrix(index_type num_rows,
                               index_type num_blocks,
                               InteractionTensorStorage storage,
                               index_container rows,
                               index_container cols,
                               value_container values)
      : num_rows_(num_rows),
        num_blocks_(num_blocks),
        storage_(storage),
        components_per_block_(interaction_tensor_storage_components(storage)),
        row_(std::move(rows)),
        col_(std::move(cols)),
        val_(std::move(values)) {
  }

  [[nodiscard]] index_type num_rows() const { return num_rows_; }
  [[nodiscard]] index_type num_blocks() const { return num_blocks_; }
  [[nodiscard]] InteractionTensorStorage storage() const { return storage_; }
  [[nodiscard]] int components_per_block() const { return components_per_block_; }
  [[nodiscard]] std::size_t memory() const { return row_.bytes() + col_.bytes() + val_.bytes(); }

  [[nodiscard]] const index_type* row_data() const { return row_.data(); }
  [[nodiscard]] const index_type* col_data() const { return col_.data(); }
  [[nodiscard]] const value_type* val_data() const { return val_.data(); }
  [[nodiscard]] Mat<value_type, 3, 3> block_tensor(index_type block) const;

  [[nodiscard]] const index_type* row_device_data() const { return row_.device_data(); }
  [[nodiscard]] const index_type* col_device_data() const { return col_.device_data(); }
  [[nodiscard]] const value_type* val_device_data() const { return val_.device_data(); }

  template<class X, class Y, size_t N>
  void multiply(const MultiArray<X, N>& spins, MultiArray<Y, N>& field) const;

  template<class X, size_t N>
  Vec<value_type, 3> multiply_row(index_type i, const MultiArray<X, N>& spins) const;

#if HAS_CUDA
  void multiply_gpu(const MultiArray<double, 2>& spins, MultiArray<value_type, 2>& field, cudaStream_t stream) const;
  void multiply_gpu(const MultiArray<float, 2>& spins, MultiArray<value_type, 2>& field, cudaStream_t stream) const;
#endif

private:
  template<class X>
  Vec<value_type, 3> multiply_row_data(index_type i, const X* spins) const;

  template<InteractionTensorStorage Storage, class X>
  Vec<value_type, 3> multiply_row_data_storage(index_type i, const X* spins) const;

  template<InteractionTensorStorage Storage, class X, class Y>
  void multiply_data_storage(const X* spins, Y* field) const;

  index_type num_rows_ = 0;
  index_type num_blocks_ = 0;
  InteractionTensorStorage storage_ = InteractionTensorStorage::Isotropic;
  int components_per_block_ = 1;
  index_container row_;
  index_container col_;
  value_container val_;
};

namespace detail {

template<typename T>
inline bool tensor_value_zero(const T value, const double tolerance) {
  return std::abs(static_cast<double>(value)) <= tolerance;
}

template<typename T>
inline bool tensor_value_equal(const T a, const T b, const double tolerance) {
  return std::abs(static_cast<double>(a - b)) <= tolerance;
}

template<typename T>
inline bool tensor_is_zero(const Mat<T, 3, 3>& value, const double tolerance) {
  for (auto m = 0; m < 3; ++m) {
    for (auto n = 0; n < 3; ++n) {
      if (!tensor_value_zero(value[m][n], tolerance)) {
        return false;
      }
    }
  }
  return true;
}

template<typename T>
inline Mat<T, 3, 3> canonical_tensor(Mat<T, 3, 3> value, const InteractionTensorStorage storage, const double tolerance) {
  if (tolerance <= 0.0) {
    return value;
  }

  switch (storage) {
    case InteractionTensorStorage::Isotropic: {
      const auto j = static_cast<T>((value[0][0] + value[1][1] + value[2][2]) / static_cast<T>(3));
      return {j, 0, 0, 0, j, 0, 0, 0, j};
    }
    case InteractionTensorStorage::Anisotropic:
      return {value[0][0], 0, 0, 0, value[1][1], 0, 0, 0, value[2][2]};
    case InteractionTensorStorage::Symmetric:
      return {value[0][0],
              static_cast<T>(0.5) * (value[0][1] + value[1][0]),
              static_cast<T>(0.5) * (value[0][2] + value[2][0]),
              static_cast<T>(0.5) * (value[1][0] + value[0][1]),
              value[1][1],
              static_cast<T>(0.5) * (value[1][2] + value[2][1]),
              static_cast<T>(0.5) * (value[2][0] + value[0][2]),
              static_cast<T>(0.5) * (value[2][1] + value[1][2]),
              value[2][2]};
    case InteractionTensorStorage::Antisymmetric:
      return {0,
              static_cast<T>(0.5) * (value[0][1] - value[1][0]),
              static_cast<T>(0.5) * (value[0][2] - value[2][0]),
              static_cast<T>(0.5) * (value[1][0] - value[0][1]),
              0,
              static_cast<T>(0.5) * (value[1][2] - value[2][1]),
              static_cast<T>(0.5) * (value[2][0] - value[0][2]),
              static_cast<T>(0.5) * (value[2][1] - value[1][2]),
              0};
    case InteractionTensorStorage::General:
    case InteractionTensorStorage::Auto:
      return value;
  }
  return value;
}

template<typename T>
inline bool tensor_fits_storage(const Mat<T, 3, 3>& value, const InteractionTensorStorage storage, const double tolerance) {
  switch (storage) {
    case InteractionTensorStorage::Auto:
      return true;
    case InteractionTensorStorage::Isotropic:
      return tensor_value_equal(value[0][0], value[1][1], tolerance)
             && tensor_value_equal(value[0][0], value[2][2], tolerance)
             && tensor_value_zero(value[0][1], tolerance)
             && tensor_value_zero(value[0][2], tolerance)
             && tensor_value_zero(value[1][0], tolerance)
             && tensor_value_zero(value[1][2], tolerance)
             && tensor_value_zero(value[2][0], tolerance)
             && tensor_value_zero(value[2][1], tolerance);
    case InteractionTensorStorage::Anisotropic:
      return tensor_value_zero(value[0][1], tolerance)
             && tensor_value_zero(value[0][2], tolerance)
             && tensor_value_zero(value[1][0], tolerance)
             && tensor_value_zero(value[1][2], tolerance)
             && tensor_value_zero(value[2][0], tolerance)
             && tensor_value_zero(value[2][1], tolerance);
    case InteractionTensorStorage::Symmetric:
      return tensor_value_equal(value[0][1], value[1][0], tolerance)
             && tensor_value_equal(value[0][2], value[2][0], tolerance)
             && tensor_value_equal(value[1][2], value[2][1], tolerance);
    case InteractionTensorStorage::Antisymmetric:
      return tensor_value_zero(value[0][0], tolerance)
             && tensor_value_zero(value[1][1], tolerance)
             && tensor_value_zero(value[2][2], tolerance)
             && tensor_value_equal(value[0][1], -value[1][0], tolerance)
             && tensor_value_equal(value[0][2], -value[2][0], tolerance)
             && tensor_value_equal(value[1][2], -value[2][1], tolerance);
    case InteractionTensorStorage::General:
      return true;
  }
  return false;
}

template<typename T>
inline InteractionTensorStorage classify_tensor_storage(const Mat<T, 3, 3>& value, const double tolerance) {
  if (tensor_fits_storage(value, InteractionTensorStorage::Isotropic, tolerance)) {
    return InteractionTensorStorage::Isotropic;
  }
  if (tensor_fits_storage(value, InteractionTensorStorage::Anisotropic, tolerance)) {
    return InteractionTensorStorage::Anisotropic;
  }
  if (tensor_fits_storage(value, InteractionTensorStorage::Antisymmetric, tolerance)) {
    return InteractionTensorStorage::Antisymmetric;
  }
  if (tensor_fits_storage(value, InteractionTensorStorage::Symmetric, tolerance)) {
    return InteractionTensorStorage::Symmetric;
  }
  return InteractionTensorStorage::General;
}

inline InteractionTensorStorage combine_tensor_storage(const InteractionTensorStorage a, const InteractionTensorStorage b) {
  if (a == InteractionTensorStorage::Auto) {
    return b;
  }
  if (b == InteractionTensorStorage::Auto) {
    return a;
  }
  if (a == b) {
    return a;
  }
  if (a == InteractionTensorStorage::General || b == InteractionTensorStorage::General) {
    return InteractionTensorStorage::General;
  }
  if (a == InteractionTensorStorage::Isotropic) {
    if (b == InteractionTensorStorage::Antisymmetric) {
      return InteractionTensorStorage::General;
    }
    return b;
  }
  if (b == InteractionTensorStorage::Isotropic) {
    if (a == InteractionTensorStorage::Antisymmetric) {
      return InteractionTensorStorage::General;
    }
    return a;
  }
  if ((a == InteractionTensorStorage::Anisotropic && b == InteractionTensorStorage::Symmetric)
      || (a == InteractionTensorStorage::Symmetric && b == InteractionTensorStorage::Anisotropic)) {
    return InteractionTensorStorage::Symmetric;
  }
  return InteractionTensorStorage::General;
}

template<typename T>
inline void pack_tensor_components(std::vector<T>& values, const InteractionTensorStorage storage, const Mat<T, 3, 3>& raw, const double tolerance) {
  const auto value = canonical_tensor(raw, storage, tolerance);
  switch (storage) {
    case InteractionTensorStorage::Isotropic:
      values.push_back(value[0][0]);
      break;
    case InteractionTensorStorage::Anisotropic:
      values.push_back(value[0][0]);
      values.push_back(value[1][1]);
      values.push_back(value[2][2]);
      break;
    case InteractionTensorStorage::Symmetric:
      values.push_back(value[0][0]);
      values.push_back(value[0][1]);
      values.push_back(value[0][2]);
      values.push_back(value[1][1]);
      values.push_back(value[1][2]);
      values.push_back(value[2][2]);
      break;
    case InteractionTensorStorage::Antisymmetric:
      values.push_back(value[0][1]);
      values.push_back(value[0][2]);
      values.push_back(value[1][2]);
      break;
    case InteractionTensorStorage::General:
      for (auto m = 0; m < 3; ++m) {
        for (auto n = 0; n < 3; ++n) {
          values.push_back(value[m][n]);
        }
      }
      break;
    case InteractionTensorStorage::Auto:
      throw std::invalid_argument("cannot pack auto tensor storage");
  }
}

template<typename T>
inline Mat<T, 3, 3> unpack_tensor_components(const T* values, const InteractionTensorStorage storage) {
  switch (storage) {
    case InteractionTensorStorage::Isotropic:
      return {values[0], 0, 0, 0, values[0], 0, 0, 0, values[0]};
    case InteractionTensorStorage::Anisotropic:
      return {values[0], 0, 0, 0, values[1], 0, 0, 0, values[2]};
    case InteractionTensorStorage::Symmetric:
      return {values[0], values[1], values[2], values[1], values[3], values[4], values[2], values[4], values[5]};
    case InteractionTensorStorage::Antisymmetric:
      return {0, values[0], values[1], -values[0], 0, values[2], -values[1], -values[2], 0};
    case InteractionTensorStorage::General:
      return {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]};
    case InteractionTensorStorage::Auto:
      throw std::invalid_argument("cannot unpack auto tensor storage");
  }
  throw std::invalid_argument("unknown interaction tensor storage");
}

template<typename T>
inline Mat<T, 3, 3> unpack_tensor_components_component_major(const T* values,
                                                             const std::size_t num_blocks,
                                                             const std::size_t block,
                                                             const InteractionTensorStorage storage) {
  switch (storage) {
    case InteractionTensorStorage::Isotropic: {
      const T xx = values[block];
      return {xx, 0, 0, 0, xx, 0, 0, 0, xx};
    }
    case InteractionTensorStorage::Anisotropic:
      return {values[block], 0, 0, 0, values[num_blocks + block], 0, 0, 0, values[2 * num_blocks + block]};
    case InteractionTensorStorage::Symmetric:
      return {values[block],
              values[num_blocks + block],
              values[2 * num_blocks + block],
              values[num_blocks + block],
              values[3 * num_blocks + block],
              values[4 * num_blocks + block],
              values[2 * num_blocks + block],
              values[4 * num_blocks + block],
              values[5 * num_blocks + block]};
    case InteractionTensorStorage::Antisymmetric:
      return {0,
              values[block],
              values[num_blocks + block],
              -values[block],
              0,
              values[2 * num_blocks + block],
              -values[num_blocks + block],
              -values[2 * num_blocks + block],
              0};
    case InteractionTensorStorage::General:
      return {values[block],
              values[num_blocks + block],
              values[2 * num_blocks + block],
              values[3 * num_blocks + block],
              values[4 * num_blocks + block],
              values[5 * num_blocks + block],
              values[6 * num_blocks + block],
              values[7 * num_blocks + block],
              values[8 * num_blocks + block]};
    case InteractionTensorStorage::Auto:
      throw std::invalid_argument("cannot unpack auto tensor storage");
  }
  throw std::invalid_argument("unknown interaction tensor storage");
}

template<typename T>
inline void multiply_tensor_components(const InteractionTensorStorage storage,
                                       const T* values,
                                       const T sx,
                                       const T sy,
                                       const T sz,
                                       T& hx,
                                       T& hy,
                                       T& hz) {
  switch (storage) {
    case InteractionTensorStorage::Isotropic:
      hx += values[0] * sx;
      hy += values[0] * sy;
      hz += values[0] * sz;
      break;
    case InteractionTensorStorage::Anisotropic:
      hx += values[0] * sx;
      hy += values[1] * sy;
      hz += values[2] * sz;
      break;
    case InteractionTensorStorage::Symmetric:
      hx += values[0] * sx + values[1] * sy + values[2] * sz;
      hy += values[1] * sx + values[3] * sy + values[4] * sz;
      hz += values[2] * sx + values[4] * sy + values[5] * sz;
      break;
    case InteractionTensorStorage::Antisymmetric:
      hx += values[0] * sy + values[1] * sz;
      hy += -values[0] * sx + values[2] * sz;
      hz += -values[1] * sx - values[2] * sy;
      break;
    case InteractionTensorStorage::General:
      hx += values[0] * sx + values[1] * sy + values[2] * sz;
      hy += values[3] * sx + values[4] * sy + values[5] * sz;
      hz += values[6] * sx + values[7] * sy + values[8] * sz;
      break;
    case InteractionTensorStorage::Auto:
      throw std::invalid_argument("cannot multiply auto tensor storage");
  }
}

template<InteractionTensorStorage Storage>
inline constexpr int tensor_storage_component_count() {
  if constexpr (Storage == InteractionTensorStorage::Isotropic) {
    return 1;
  } else if constexpr (Storage == InteractionTensorStorage::Anisotropic) {
    return 3;
  } else if constexpr (Storage == InteractionTensorStorage::Symmetric) {
    return 6;
  } else if constexpr (Storage == InteractionTensorStorage::Antisymmetric) {
    return 3;
  } else {
    return 9;
  }
}

}  // namespace detail

template<typename T>
Mat<T, 3, 3> BlockSparseInteractionMatrix<T>::block_tensor(
    const typename BlockSparseInteractionMatrix<T>::index_type block) const {
  if (block < 0 || block >= num_blocks_) {
    throw std::runtime_error("Invalid block index for block sparse interaction matrix");
  }
  return detail::unpack_tensor_components_component_major(
      val_.data(),
      static_cast<std::size_t>(num_blocks_),
      static_cast<std::size_t>(block),
      storage_);
}

template<typename T>
class BlockSparseInteractionMatrix<T>::Builder {
public:
  Builder() = default;

  Builder(index_type num_rows, InteractionTensorStorage requested_storage, double tolerance)
      : requested_storage_(requested_storage),
        storage_(requested_storage == InteractionTensorStorage::Auto ? InteractionTensorStorage::Isotropic : requested_storage),
        tolerance_(tolerance),
        num_rows_(num_rows) {
  }

  void insert(index_type i, index_type j, const value_type& value) {
    insert(i, j, value * kIdentityMat3R);
  }

  void insert(index_type i, index_type j, const Mat<value_type, 3, 3>& value) {
    assert_index_is_valid(i, j);
    if (detail::tensor_is_zero(value, 0.0)) {
      return;
    }

    const auto value_storage = detail::classify_tensor_storage(value, tolerance_);
    if (requested_storage_ == InteractionTensorStorage::Auto) {
      const auto next_storage = row_.empty()
          ? value_storage
          : detail::combine_tensor_storage(storage_, value_storage);
      if (next_storage != storage_) {
        repack(next_storage);
      }
    } else if (!detail::tensor_fits_storage(value, requested_storage_, tolerance_)) {
      throw std::runtime_error("interaction tensor does not fit requested tensor_storage=" + to_string(requested_storage_)
                               + " for sites " + std::to_string(i) + ", " + std::to_string(j));
    }

    row_.push_back(i);
    col_.push_back(j);
    detail::pack_tensor_components(val_, storage_, value, tolerance_);
    is_sorted_ = false;
    is_merged_ = false;
  }

  [[nodiscard]] std::size_t memory() const {
    return row_.capacity() * sizeof(index_type)
           + col_.capacity() * sizeof(index_type)
           + val_.capacity() * sizeof(value_type);
  }

  [[nodiscard]] InteractionTensorStorage storage() const {
    return storage_;
  }

  void output(std::ostream& os) {
    sort();
    merge();
    prune_zero_blocks();
    const int components = component_count();
    for (std::size_t block = 0; block < row_.size(); ++block) {
      const auto value = detail::unpack_tensor_components(val_.data() + block * components, storage_);
      for (auto m = 0; m < 3; ++m) {
        for (auto n = 0; n < 3; ++n) {
          if (value[m][n] != value_type{}) {
            os << (3 * row_[block] + m) << " " << (3 * col_[block] + n) << " " << value[m][n] << "\n";
          }
        }
      }
    }
  }

  bool is_structurally_symmetric() {
    sort();
    merge();
    prune_zero_blocks();

    for (std::size_t n = 0; n < row_.size(); ++n) {
      const auto i = row_[n];
      const auto j = col_[n];
      if (!find_block(j, i)) {
        return false;
      }
    }
    return true;
  }

  bool is_symmetric() {
    sort();
    merge();
    prune_zero_blocks();

    const int components = component_count();
    for (std::size_t n = 0; n < row_.size(); ++n) {
      const auto i = row_[n];
      const auto j = col_[n];
      const auto ji = find_block(j, i);
      if (!ji) {
        return false;
      }

      const auto ij_value = detail::unpack_tensor_components(val_.data() + n * components, storage_);
      const auto ji_value = detail::unpack_tensor_components(val_.data() + *ji * components, storage_);
      for (auto m = 0; m < 3; ++m) {
        for (auto p = 0; p < 3; ++p) {
          if (ij_value[m][p] != ji_value[p][m]) {
            return false;
          }
        }
      }
    }
    return true;
  }

  BlockSparseInteractionMatrix<T> build() {
    sort();
    merge();
    prune_zero_blocks();

    const auto num_blocks = checked_index(row_.size(), "number of block sparse interactions");
    index_container csr_rows(num_rows_ + 1);
    csr_rows(0) = 0;

    index_type current_row = 0;
    index_type previous_row = 0;
    for (std::size_t n = 0; n < row_.size(); ++n) {
      current_row = row_[n];
      if (current_row == previous_row) {
        continue;
      }
      for (auto i = previous_row + 1; i < current_row + 1; ++i) {
        csr_rows(i) = checked_index(n, "CSR row offset");
      }
      previous_row = current_row;
    }
    for (auto i = previous_row + 1; i < num_rows_ + 1; ++i) {
      csr_rows(i) = num_blocks;
    }

    index_container csr_cols(col_.begin(), col_.end());
    std::vector<value_type> component_major_values(row_.size() * component_count());
    const int components = component_count();
    const auto blocks = row_.size();
    for (std::size_t block = 0; block < blocks; ++block) {
      for (auto c = 0; c < components; ++c) {
        component_major_values[c * blocks + block] = val_[block * components + c];
      }
    }

    value_container csr_vals(component_major_values.begin(), component_major_values.end());

    clear();
    return BlockSparseInteractionMatrix<T>(num_rows_, num_blocks, storage_, std::move(csr_rows), std::move(csr_cols), std::move(csr_vals));
  }

  void clear() {
    util::force_deallocation(row_);
    util::force_deallocation(col_);
    util::force_deallocation(val_);
    is_sorted_ = false;
    is_merged_ = false;
  }

private:
  [[nodiscard]] int component_count() const {
    return interaction_tensor_storage_components(storage_);
  }

  static index_type checked_index(const std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<index_type>::max())) {
      throw std::runtime_error(std::string(label) + " exceeds block sparse index_type");
    }
    return static_cast<index_type>(value);
  }

  void assert_index_is_valid(index_type i, index_type j) const {
    if ((i >= num_rows_) || (i < 0) || (j >= num_rows_) || (j < 0)) {
      throw std::runtime_error("Invalid index for block sparse interaction matrix");
    }
  }

  void repack(const InteractionTensorStorage next_storage) {
    const int old_components = component_count();
    std::vector<value_type> next_values;
    next_values.reserve(row_.size() * interaction_tensor_storage_components(next_storage));
    for (std::size_t block = 0; block < row_.size(); ++block) {
      const auto value = detail::unpack_tensor_components(val_.data() + block * old_components, storage_);
      detail::pack_tensor_components(next_values, next_storage, value, tolerance_);
    }
    storage_ = next_storage;
    val_.swap(next_values);
  }

  void sort() {
    if (is_sorted_) {
      return;
    }

    const int components = component_count();
    std::vector<std::size_t> permutation(row_.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&](std::size_t a, std::size_t b) {
      if (row_[a] < row_[b]) {
        return true;
      }
      if (row_[a] == row_[b]) {
        return col_[a] < col_[b];
      }
      return false;
    });

    std::vector<index_type> sorted_rows(row_.size());
    std::vector<index_type> sorted_cols(col_.size());
    std::vector<value_type> sorted_vals(val_.size());
    for (std::size_t out = 0; out < permutation.size(); ++out) {
      const auto in = permutation[out];
      sorted_rows[out] = row_[in];
      sorted_cols[out] = col_[in];
      for (auto c = 0; c < components; ++c) {
        sorted_vals[out * components + c] = val_[in * components + c];
      }
    }

    row_.swap(sorted_rows);
    col_.swap(sorted_cols);
    val_.swap(sorted_vals);
    is_sorted_ = true;
  }

  void merge() {
    if (is_merged_) {
      return;
    }
    if (row_.empty()) {
      is_merged_ = true;
      return;
    }

    const int components = component_count();
    std::size_t write = 0;
    for (std::size_t read = 1; read < row_.size(); ++read) {
      if (row_[read] == row_[write] && col_[read] == col_[write]) {
        for (auto c = 0; c < components; ++c) {
          val_[write * components + c] += val_[read * components + c];
        }
        continue;
      }
      ++write;
      if (write != read) {
        row_[write] = row_[read];
        col_[write] = col_[read];
        for (auto c = 0; c < components; ++c) {
          val_[write * components + c] = val_[read * components + c];
        }
      }
    }
    resize_blocks(write + 1);
    is_merged_ = true;
  }

  void prune_zero_blocks() {
    if (row_.empty()) {
      return;
    }

    const int components = component_count();
    std::size_t write = 0;
    for (std::size_t read = 0; read < row_.size(); ++read) {
      bool zero = true;
      for (auto c = 0; c < components; ++c) {
        if (val_[read * components + c] != value_type{}) {
          zero = false;
          break;
        }
      }
      if (zero) {
        continue;
      }
      if (write != read) {
        row_[write] = row_[read];
        col_[write] = col_[read];
        for (auto c = 0; c < components; ++c) {
          val_[write * components + c] = val_[read * components + c];
        }
      }
      ++write;
    }
    resize_blocks(write);
  }

  void resize_blocks(const std::size_t blocks) {
    const int components = component_count();
    row_.resize(blocks);
    col_.resize(blocks);
    val_.resize(blocks * components);
  }

  [[nodiscard]] std::optional<std::size_t> find_block(index_type row, index_type col) const {
    const auto row_begin = std::lower_bound(row_.cbegin(), row_.cend(), row);
    if (row_begin == row_.cend() || *row_begin != row) {
      return std::nullopt;
    }
    const auto row_end = std::upper_bound(row_begin, row_.cend(), row);
    const auto begin_index = static_cast<std::size_t>(row_begin - row_.cbegin());
    const auto end_index = static_cast<std::size_t>(row_end - row_.cbegin());
    const auto col_begin = col_.cbegin() + static_cast<std::ptrdiff_t>(begin_index);
    const auto col_end = col_.cbegin() + static_cast<std::ptrdiff_t>(end_index);
    const auto col_it = std::lower_bound(col_begin, col_end, col);
    if (col_it == col_end || *col_it != col) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(col_it - col_.cbegin());
  }

  InteractionTensorStorage requested_storage_ = InteractionTensorStorage::Auto;
  InteractionTensorStorage storage_ = InteractionTensorStorage::Isotropic;
  double tolerance_ = 0.0;
  index_type num_rows_ = 0;
  bool is_sorted_ = false;
  bool is_merged_ = false;
  std::vector<index_type> row_;
  std::vector<index_type> col_;
  std::vector<value_type> val_;
};

template<typename T>
template<class X>
Vec<T, 3> BlockSparseInteractionMatrix<T>::multiply_row_data(const index_type i, const X* spins) const {
  switch (storage_) {
    case InteractionTensorStorage::Isotropic:
      return multiply_row_data_storage<InteractionTensorStorage::Isotropic>(i, spins);
    case InteractionTensorStorage::Anisotropic:
      return multiply_row_data_storage<InteractionTensorStorage::Anisotropic>(i, spins);
    case InteractionTensorStorage::Symmetric:
      return multiply_row_data_storage<InteractionTensorStorage::Symmetric>(i, spins);
    case InteractionTensorStorage::Antisymmetric:
      return multiply_row_data_storage<InteractionTensorStorage::Antisymmetric>(i, spins);
    case InteractionTensorStorage::General:
      return multiply_row_data_storage<InteractionTensorStorage::General>(i, spins);
    case InteractionTensorStorage::Auto:
      throw std::runtime_error("cannot multiply auto tensor storage");
  }
  throw std::runtime_error("unknown interaction tensor storage");
}

template<typename T>
template<InteractionTensorStorage Storage, class X>
Vec<T, 3> BlockSparseInteractionMatrix<T>::multiply_row_data_storage(const index_type i, const X* spins) const {
  const T* values = val_.data();
  const auto blocks = num_blocks_;
  T hx = 0;
  T hy = 0;
  T hz = 0;
  for (auto n = row_(i); n < row_(i + 1); ++n) {
    const auto j = col_(n);
    const auto base = 3 * j;
    const auto sx = static_cast<T>(spins[base + 0]);
    const auto sy = static_cast<T>(spins[base + 1]);
    const auto sz = static_cast<T>(spins[base + 2]);

    if constexpr (Storage == InteractionTensorStorage::Isotropic) {
      const T j0 = values[n];
      hx += j0 * sx;
      hy += j0 * sy;
      hz += j0 * sz;
    } else if constexpr (Storage == InteractionTensorStorage::Anisotropic) {
      hx += values[n] * sx;
      hy += values[blocks + n] * sy;
      hz += values[2 * blocks + n] * sz;
    } else if constexpr (Storage == InteractionTensorStorage::Symmetric) {
      hx += values[n] * sx + values[blocks + n] * sy + values[2 * blocks + n] * sz;
      hy += values[blocks + n] * sx + values[3 * blocks + n] * sy + values[4 * blocks + n] * sz;
      hz += values[2 * blocks + n] * sx + values[4 * blocks + n] * sy + values[5 * blocks + n] * sz;
    } else if constexpr (Storage == InteractionTensorStorage::Antisymmetric) {
      hx += values[n] * sy + values[blocks + n] * sz;
      hy += -values[n] * sx + values[2 * blocks + n] * sz;
      hz += -values[blocks + n] * sx - values[2 * blocks + n] * sy;
    } else {
      hx += values[n] * sx + values[blocks + n] * sy + values[2 * blocks + n] * sz;
      hy += values[3 * blocks + n] * sx + values[4 * blocks + n] * sy + values[5 * blocks + n] * sz;
      hz += values[6 * blocks + n] * sx + values[7 * blocks + n] * sy + values[8 * blocks + n] * sz;
    }
  }
  return {hx, hy, hz};
}

template<typename T>
template<class X, size_t N>
Vec<T, 3> BlockSparseInteractionMatrix<T>::multiply_row(const index_type i, const MultiArray<X, N>& spins) const {
  return multiply_row_data(i, spins.data());
}

template<typename T>
template<class X, class Y, size_t N>
void BlockSparseInteractionMatrix<T>::multiply(const MultiArray<X, N>& spins, MultiArray<Y, N>& field) const {
  const auto* spin_data = spins.data();
  auto* field_data = field.data();
  switch (storage_) {
    case InteractionTensorStorage::Isotropic:
      multiply_data_storage<InteractionTensorStorage::Isotropic>(spin_data, field_data);
      return;
    case InteractionTensorStorage::Anisotropic:
      multiply_data_storage<InteractionTensorStorage::Anisotropic>(spin_data, field_data);
      return;
    case InteractionTensorStorage::Symmetric:
      multiply_data_storage<InteractionTensorStorage::Symmetric>(spin_data, field_data);
      return;
    case InteractionTensorStorage::Antisymmetric:
      multiply_data_storage<InteractionTensorStorage::Antisymmetric>(spin_data, field_data);
      return;
    case InteractionTensorStorage::General:
      multiply_data_storage<InteractionTensorStorage::General>(spin_data, field_data);
      return;
    case InteractionTensorStorage::Auto:
      throw std::runtime_error("cannot multiply auto tensor storage");
  }
  throw std::runtime_error("unknown interaction tensor storage");
}

template<typename T>
template<InteractionTensorStorage Storage, class X, class Y>
void BlockSparseInteractionMatrix<T>::multiply_data_storage(const X* spins, Y* field) const {
#if HAS_OMP
#pragma omp parallel for
#endif
  for (index_type i = 0; i < num_rows_; ++i) {
    const auto h = multiply_row_data_storage<Storage>(i, spins);
    const auto base = 3 * i;
    field[base + 0] = static_cast<Y>(h[0]);
    field[base + 1] = static_cast<Y>(h[1]);
    field[base + 2] = static_cast<Y>(h[2]);
  }
}

}  // namespace jams

#endif  // JAMS_CONTAINERS_BLOCK_SPARSE_INTERACTION_MATRIX_H
