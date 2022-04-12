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
#include <jams/cuda/cuda_common.h>
#include <jams/cuda/cuda_types.h>
#endif

// The cuSPARSE generic API was added in CUDA 10 although appears to be
// incomplete in 10.1.105. So we use the new API only for versions 10.2 and
// greater. From CUDA 11 the 'cusparse<X>csrmv' functions have been removed
// so we MUST use the new API.
#if defined(HAS_CUDA) && CUDART_VERSION >= 10012
#define HAS_CUSPARSE_GENERIC_API 1
#else
#define HAS_CUSPARSE_GENERIC_API 0
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
        using index_type             = int32_t;
        using index_reference        = index_type &;
        using const_index_reference  = const index_type &;
        using index_pointer          = index_type *;
        using const_size_pointer    = const index_type *;
        using index_container = MultiArray<index_type, 1>;
        using value_container = MultiArray<value_type, 1>;

        SparseMatrix() = default;

        inline SparseMatrix(const index_type num_rows, const index_type num_cols, const index_type num_non_zero,
                            index_container rows, index_container cols, value_container vals,
                            SparseMatrixFormat format, SparseMatrixType type, SparseMatrixFillMode fill_mode)
            : description_(format, type, fill_mode, SparseMatrixDiagType::NON_UNIT),
              num_rows_(num_rows),
              num_cols_(num_cols),
              num_non_zero_(num_non_zero),
              row_(std::move(rows)),
              col_(std::move(cols)),
              val_(std::move(vals)) {
        }

        inline ~SparseMatrix() {
          #if HAS_CUSPARSE_GENERIC_API
          if (cusparse_sp_mat_descr_) {
            cusparseDestroySpMat(cusparse_sp_mat_descr_);
            cusparse_sp_mat_descr_ = nullptr;
          }

          if (vector_dn_vec_descr_) {
            cusparseDestroyDnVec(vector_dn_vec_descr_);
            vector_dn_vec_descr_ = nullptr;
          }

          if (vector_dn_vec_descr_) {
            cusparseDestroyDnVec(result_dn_vec_descr_);
            result_dn_vec_descr_ = nullptr;
          }

          if (cusparse_buffer_) {
            cudaFree(cusparse_buffer_);
            cusparse_buffer_ = nullptr;
            cusparse_buffer_size_ = 0;
          }
          #endif
        }

        inline constexpr SparseMatrixFormat format() const { return description_.format(); }

        inline constexpr SparseMatrixType type() const { return description_.type(); }

        inline constexpr SparseMatrixFillMode fill_mode() const { return description_.fill_mode(); }

        inline constexpr index_type num_non_zero() const { return num_non_zero_; }

        inline constexpr index_type num_rows() const { return num_rows_; }

        inline constexpr index_type num_cols() const { return num_cols_; }

        inline const_size_pointer row_data()  const { return row_.data(); }
        inline const_size_pointer col_data()  const { return col_.data(); }
        inline const_value_pointer val_data()  const { return val_.data(); }

        inline const_size_pointer row_device_data()  const { return row_.device_data(); }
        inline const_size_pointer col_device_data()  const { return col_.device_data(); }
        inline const_value_pointer val_device_data()  const { return val_.device_data(); }

        inline constexpr std::size_t memory() const { return row_.memory() + col_.memory() + val_.memory(); };

        template<class U, size_t N>
        void multiply(const MultiArray<U, N> &vector, MultiArray<U, N> &result) const;

        template<class U, size_t N>
        U multiply_row(index_type i, const MultiArray<U, N> &vector) const;

        #if HAS_CUDA
        template <class U, size_t N>
        void multiply_gpu(const MultiArray<U, N> &vector, MultiArray<U, N> &result,
                          cusparseHandle_t handle, cudaStream_t stream_id);
        #endif

    protected:
        SparseMatrixDescription description_;

        #if HAS_CUSPARSE_GENERIC_API
        cusparseSpMatDescr_t cusparse_sp_mat_descr_ = nullptr;

        // descriptors for dense vectors used in matrix multiplication
        cusparseDnVecDescr_t vector_dn_vec_descr_ = nullptr;
        cusparseDnVecDescr_t result_dn_vec_descr_ = nullptr;

        void*       cusparse_buffer_ = nullptr;
        std::size_t cusparse_buffer_size_ = 0;
        #endif

        index_type num_rows_             = 0;
        index_type num_cols_             = 0;
        index_type num_non_zero_         = 0;
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
          return;
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
          return;
      }
      throw std::runtime_error("Unknown sparse matrix format for SparseMatrix<T>::multiply");
    }

    template<typename T>
    template<class U, size_t N>
    U SparseMatrix<T>::multiply_row(const index_type i, const MultiArray<U, N> &vector) const {
      switch (description_.format()) {
        case SparseMatrixFormat::COO:
          return jams::Xcoomv_general_row(
              val_.data(), col_.data(), row_.data(), vector.data(), i);
        case SparseMatrixFormat::CSR:
          return jams::Xcsrmv_general_row(
              val_.data(), col_.data(), row_.data(), vector.data(), i);
      }
      throw std::runtime_error("Unknown sparse matrix format for SparseMatrix<T>::multiply_row");
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
          const T one = 1.0, zero = 0.0;

    #if HAS_CUSPARSE_GENERIC_API

          if (!cusparse_sp_mat_descr_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateCsr(
                &cusparse_sp_mat_descr_,
                num_rows_,
                num_cols_,
                num_non_zero_,
                row_.device_data(),
                col_.device_data(),
                val_.device_data(),
                cuda::get_cusparse_index_type<index_type>(),
                cuda::get_cusparse_index_type<index_type>(),
                CUSPARSE_INDEX_BASE_ZERO,
                cuda::get_cuda_data_type<T>()
            ));
          }

          if (!vector_dn_vec_descr_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateDnVec(
                &vector_dn_vec_descr_,
                vector.elements(),
                (void*)vector.device_data(),
                cuda::get_cuda_data_type<U>()));
          }
          cusparseDnVecSetValues(vector_dn_vec_descr_, (void*)vector.device_data());

          if (!result_dn_vec_descr_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateDnVec(
                &result_dn_vec_descr_,
                result.elements(),
                (void*)result.device_data(),
                cuda::get_cuda_data_type<U>()));
          }
          cusparseDnVecSetValues(result_dn_vec_descr_, (void*)result.device_data());

          size_t new_buffer_size = 0;

          CHECK_CUSPARSE_STATUS(cusparseSpMV_bufferSize(
              handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &one,
              cusparse_sp_mat_descr_,
              vector_dn_vec_descr_,
              &zero,
              result_dn_vec_descr_,
              // Note: the type selection here may be more complicated in general.
              // The compute type depends on the types of A/X/Y.
              // see https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmv
              cuda::get_cuda_data_type<U>(),
              CUSPARSE_SPMV_CSR_ALG1,
              &new_buffer_size));

          if (new_buffer_size > cusparse_buffer_size_) {
            if (cusparse_buffer_) {
              cudaFree(cusparse_buffer_);
            }
            cudaMalloc(&cusparse_buffer_, new_buffer_size);
            cusparse_buffer_size_ = new_buffer_size;
          }

          CHECK_CUSPARSE_STATUS(cusparseSpMV(
              handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &one,
              cusparse_sp_mat_descr_,
              vector_dn_vec_descr_,
              &zero,
              result_dn_vec_descr_,
              // Note: the type selection here may be more complex in general.
              // The compute type depends on the types of A/X/Y.
              // see (https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmv)
              cuda::get_cuda_data_type<U>(),
              CUSPARSE_SPMV_CSR_ALG1,
              cusparse_buffer_));
    #else
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
    #endif
          cusparseSetStream(handle, 0);
          break;

      }
    }


    #endif



}

#undef HAS_CUSPARSE_GENERIC_API

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_H
