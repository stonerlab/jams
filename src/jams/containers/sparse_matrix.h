// Copyright 2014 Joseph Barker. All rights reserved.
#ifndef JAMS_CONTAINERS_SPARSE_MATRIX_H
#define JAMS_CONTAINERS_SPARSE_MATRIX_H

#include <cstdint>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "jams/containers/sparse_matrix_description.h"
#include "jams/interface/sparse_blas.h"
#include "jams/containers/multiarray.h"
#include "jams/helpers/exception.h"
#include "jams/helpers/sort.h"

#if HAS_MKL
#include <mkl_spblas.h>
#endif

#ifdef HAS_CUDA
#include <cusparse.h>
#include "jams/cuda/cuda_common.h"
#endif


namespace jams {
    template<typename T>
    class SparseMatrix {
    public:
        class Builder;

        using value_type            = T;
        using value_reference       = value_type &;
        using const_value_reference = const value_type &;
        using value_pointer         = value_type *;
        using const_value_pointer   = const value_type *;
        using size_type             = int;
        using size_reference        = size_type &;
        using const_size_reference  = const size_type &;
        using size_pointer          = size_type *;
        using const_size_pointer    = const size_type *;
        using index_container = MultiArray<size_type, 1>;
        using value_container = MultiArray<value_type, 1>;

        SparseMatrix() = default;

        SparseMatrix(const size_type num_rows, const size_type num_cols, const size_type num_non_zero,
                     index_container rows, index_container cols, value_container vals,
                     SparseMatrixFormat format, SparseMatrixType type, SparseMatrixFillMode fill_mode)
            : description_(format, type, fill_mode, SparseMatrixDiagType::NON_UNIT),
              num_rows_(num_rows),
              num_cols_(num_cols),
              num_non_zero_(num_non_zero),
              row_(std::move(rows)),
              col_(std::move(cols)),
              val_(std::move(vals)) {}

        inline constexpr SparseMatrixFormat format() const { return description_.format(); }

        inline constexpr SparseMatrixType type() const { return description_.type(); }

        inline constexpr SparseMatrixFillMode fill_mode() const { return description_.fill_mode(); }

        inline constexpr size_type num_non_zero() const { return num_non_zero_; }

        inline constexpr size_type num_rows() const { return num_rows_; }

        inline constexpr size_type num_cols() const { return num_cols_; }

        inline const_size_pointer row_data()  const { row_.data(); }
        inline const_size_pointer col_data()  const { col_.data(); }
        inline const_value_pointer val_data()  const { val_.data(); }

        inline const_size_pointer row_device_data()  const { row_.device_data(); }
        inline const_size_pointer col_device_data()  const { col_.device_data(); }
        inline const_value_pointer val_device_data()  const { val_.device_data(); }

        inline constexpr std::size_t memory() const { return row_.memory() + col_.memory() + val_.memory(); };

        template<class U, size_t N>
        void multiply(const MultiArray<U, N> &vector, MultiArray<U, N> &result) const;

        template<class U, size_t N>
        U multiply_row(size_type i, const MultiArray<U, N> &vector) const;

        #if HAS_CUDA
        template <class U, size_t N>
        void multiply_gpu(const MultiArray<U, N> &vector, MultiArray<U, N> &result,
                          cusparseHandle_t handle, cudaStream_t stream_id);
        #endif

    protected:
          SparseMatrixDescription description_;

        size_type num_rows_             = 0;
        size_type num_cols_             = 0;
        size_type num_non_zero_         = 0;
        index_container row_;
        index_container col_;
        value_container val_;
    };

    template<typename T>
    template<class U, size_t N>
    void SparseMatrix<T>::multiply(const MultiArray<U, N> &vector, MultiArray<U, N> &result) const {
      switch (description_.format()) {
        case SparseMatrixFormat::COO:
          jams::Xcoomv_general(1.0, 0.0, num_rows_, num_non_zero_, val_.data(), col_.data(), row_.data(),
              vector.data(), result.data());
        case SparseMatrixFormat::CSR:
          #if HAS_MKL
          double one = 1.0, zero = 0.0;
          const char transa[1] = {'N'};
          mkl_dcsrmv(transa, &num_rows_, &num_cols_, &one, description_.mkl_desc(), val_.data(),
                     col_.data(), row_.data(), row_.data() + 1, vector.data(), &zero, result.data());
          #else
          jams::Xcsrmv_general(
              1.0, 0.0, num_rows_, val_.data(), col_.data(), row_.data(), vector.data(), result.data());
          #endif
      }
    }

    template<typename T>
    template<class U, size_t N>
    U SparseMatrix<T>::multiply_row(const size_type i, const MultiArray<U, N> &vector) const {
      switch (description_.format()) {
        case SparseMatrixFormat::COO:
          return jams::Xcoomv_general_row(
              val_.data(), col_.data(), row_.data(), vector.data(), i);
        case SparseMatrixFormat::CSR:
          return jams::Xcsrmv_general_row(
              val_.data(), col_.data(), row_.data(), vector.data(), i);
      }
    }

    #ifdef HAS_CUDA
    template<typename T>
    template<class U, size_t N>
    void SparseMatrix<T>::multiply_gpu(const MultiArray<U, N> &vector, MultiArray<U, N> &result,
                                       cusparseHandle_t handle, cudaStream_t stream_id) {
      assert(handle != nullptr);
      switch (description_.format()) {
        case SparseMatrixFormat::COO:
          throw std::runtime_error("unimplemented");
        case SparseMatrixFormat::CSR:
          cusparseSetStream(handle, stream_id);
          // TODO: implement CUDA 10 generic API version remember to (static)assert types are all the same if CUDA < 10
          const T one = 1.0, zero = 0.0;
          CHECK_CUSPARSE_STATUS(cusparseDcsrmv(
             handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             num_rows_, num_cols_, num_non_zero_,
             &one,
             description_.cusparse_desc(),
             val_.device_data(), row_.device_data(), col_.device_data(),
             vector.device_data(),
             &zero,
             result.device_data()));
          cusparseSetStream(handle, 0);
          break;
      }
    }
    #endif

}

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_H
