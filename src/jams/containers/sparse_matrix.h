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
#include <mkl_version.h>
#endif

// Since MKL >= 2018 the NIST sparse BLAS interface is deprecated in MKL and
// replaced with the new 'Inspector Executor' API. This macro is '1' if
// the Inspector Executor API is available.
#if defined(HAS_MKL) && INTEL_MKL_VERSION >= 20180000
#define HAS_MKL_INSPECTOR_EXECUTOR_API 1
#else
#define HAS_MKL_INSPECTOR_EXECUTOR_API 0
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
            : matrix_A_description_(format, type, fill_mode, SparseMatrixDiagType::NON_UNIT),
              num_rows_(num_rows),
              num_cols_(num_cols),
              num_non_zero_(num_non_zero),
              row_(std::move(rows)),
              col_(std::move(cols)),
              val_(std::move(vals)) {
        }

        inline ~SparseMatrix() {
          #if HAS_MKL_INSPECTOR_EXECUTOR_API
          if (mkl_matrix_A_handle_) {
            mkl_sparse_destroy(mkl_matrix_A_handle_);
            mkl_matrix_A_handle_ = nullptr;
          }
          #endif

          #if HAS_CUSPARSE_GENERIC_API
          if (cusparse_matrix_A_handle_) {
            cusparseDestroySpMat(cusparse_matrix_A_handle_);
            cusparse_matrix_A_handle_ = nullptr;
          }

          if (cusparse_vector_x_handle_) {
            cusparseDestroyDnVec(cusparse_vector_x_handle_);
            cusparse_vector_x_handle_ = nullptr;
          }

          if (cusparse_vector_x_handle_) {
            cusparseDestroyDnVec(cusparse_vector_y_handle_);
            cusparse_vector_y_handle_ = nullptr;
          }

          if (cusparse_buffer_) {
            cudaFree(cusparse_buffer_);
            cusparse_buffer_ = nullptr;
            cusparse_buffer_size_ = 0;
          }
          #endif
        }

        inline constexpr SparseMatrixFormat format() const { return matrix_A_description_.format(); }

        inline constexpr SparseMatrixType type() const { return matrix_A_description_.type(); }

        inline constexpr SparseMatrixFillMode fill_mode() const { return matrix_A_description_.fill_mode(); }

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

        // Performs the multiplication y = A * x where 'A' is this sparse matrix
        // and vectors x and y are dense vectors passed into the function.
        template<class U, size_t N>
        void multiply(const MultiArray<U, N> &vector_x, MultiArray<U, N> &vector_y);


        // Multiplies row 'i' of this sparse matrix by the dense vector x
        // producing a single scalar.
        template<class U, size_t N>
        U multiply_row(index_type i, const MultiArray<U, N> &vector_x) const;

        #if HAS_CUDA
        // Performs the multiplication y = A * x where 'A' is this sparse matrix
        // and vectors x and y are dense vectors passed into the function.
        // Thus function performs the multiplication on the GPU.
        template <class U, size_t N>
        void multiply_gpu(const MultiArray<U, N> &vector_x, MultiArray<U, N> &vector_y,
                          cusparseHandle_t handle, cudaStream_t stream_id);
        #endif

    protected:
        SparseMatrixDescription matrix_A_description_;

        #if HAS_MKL_INSPECTOR_EXECUTOR_API
        sparse_matrix_t mkl_matrix_A_handle_ = nullptr;
        #endif

        #if HAS_CUSPARSE_GENERIC_API
        cusparseSpMatDescr_t cusparse_matrix_A_handle_ = nullptr;

        // descriptors for dense vectors used in matrix multiplication
        cusparseDnVecDescr_t cusparse_vector_x_handle_ = nullptr;
        cusparseDnVecDescr_t cusparse_vector_y_handle_ = nullptr;

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
    void SparseMatrix<T>::multiply(const MultiArray<U, N> &vector_x, MultiArray<U, N> &vector_y) {
      switch (matrix_A_description_.format()) {
        case SparseMatrixFormat::COO:
          jams::Xcoomv_general(1.0, 0.0, num_rows_, num_non_zero_, val_.data(), col_.data(), row_.data(),
              vector_x.data(), vector_y.data());
          return;
        case SparseMatrixFormat::CSR:
          #if HAS_MKL
            // Since MKL >= 2018 the NIST sparse BLAS interface is being
            // deprecated in MKL and replaced with the new 'Inspector Executor'
            // API. This requires creating a sparse matrix handle which contains
            // basic information about the matrix. We do this on the first
            // call of this function and clean it up in the destructor.
            //
            // Because this matrix is constructed by a factory the internal
            // structure of the matrix should not change so we don't need to
            // keep checking if the handle needs updating.
            //
            // We retain the old interface for older versions of MKL.
            #if HAS_MKL_INSPECTOR_EXECUTOR_API
              if (!mkl_matrix_A_handle_) {
                mkl_sparse_d_create_csr(&mkl_matrix_A_handle_,
                                        SPARSE_INDEX_BASE_ZERO,
                                        num_rows_, num_cols_,
                                        row_.data(), row_.data() + 1,
                                        col_.data(), val_.data());
              }
              mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                              1.0, mkl_matrix_A_handle_,
                              matrix_A_description_.mkl_inspector_executor_desc(),
                              vector_x.data(), 0.0, vector_y.data());
            #else
              double one = 1.0, zero = 0.0;
              const char transa[1] = {'N'};
              mkl_dcsrmv(transa, &num_rows_, &num_cols_, &one, matrix_A_description_.mkl_desc(), val_.data(),
                         col_.data(), row_.data(), row_.data() + 1, vector_x.data(), &zero, vector_y.data());
            #endif
          #else
          jams::Xcsrmv_general(
              1.0, 0.0, num_rows_, val_.data(), col_.data(), row_.data(), vector_x.data(), vector_y.data());
          #endif
          return;
      }
      throw std::runtime_error("Unknown sparse matrix format for SparseMatrix<T>::multiply");
    }

    template<typename T>
    template<class U, size_t N>
    U SparseMatrix<T>::multiply_row(const index_type i, const MultiArray<U, N> &vector_x) const {
      switch (matrix_A_description_.format()) {
        case SparseMatrixFormat::COO:
          return jams::Xcoomv_general_row(
              val_.data(), col_.data(), row_.data(), vector_x.data(), i);
        case SparseMatrixFormat::CSR:
          return jams::Xcsrmv_general_row(
              val_.data(), col_.data(), row_.data(), vector_x.data(), i);
      }
      throw std::runtime_error("Unknown sparse matrix format for SparseMatrix<T>::multiply_row");
    }

    #ifdef HAS_CUDA
    template<typename T>
    template<class U, size_t N>
    void SparseMatrix<T>::multiply_gpu(const MultiArray<U, N> &vector_x, MultiArray<U, N> &vector_y,
                                       cusparseHandle_t handle, cudaStream_t stream_id) {
      assert(handle != nullptr);
      switch (matrix_A_description_.format()) {
        case SparseMatrixFormat::COO:
          throw std::runtime_error("unimplemented");
        case SparseMatrixFormat::CSR:
          cusparseSetStream(handle, stream_id);
          const T one = 1.0, zero = 0.0;

    #if HAS_CUSPARSE_GENERIC_API

          if (!cusparse_matrix_A_handle_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateCsr(
                &cusparse_matrix_A_handle_,
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

          if (!cusparse_vector_x_handle_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateDnVec(
                &cusparse_vector_x_handle_,
                vector_x.elements(),
                (void*)vector_x.device_data(),
                cuda::get_cuda_data_type<U>()));
          }
          cusparseDnVecSetValues(cusparse_vector_x_handle_, (void*)vector_x.device_data());

          if (!cusparse_vector_y_handle_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateDnVec(
                &cusparse_vector_y_handle_,
                vector_y.elements(),
                (void*)vector_y.device_data(),
                cuda::get_cuda_data_type<U>()));
          }
          cusparseDnVecSetValues(cusparse_vector_y_handle_, (void*)vector_y.device_data());

          size_t new_buffer_size = 0;

          CHECK_CUSPARSE_STATUS(cusparseSpMV_bufferSize(
              handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &one,
              cusparse_matrix_A_handle_,
              cusparse_vector_x_handle_,
              &zero,
              cusparse_vector_y_handle_,
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
              cusparse_matrix_A_handle_,
              cusparse_vector_x_handle_,
              &zero,
              cusparse_vector_y_handle_,
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
             matrix_A_description_.cusparse_desc(),
             val_.device_data(), row_.device_data(), col_.device_data(),
             vector_x.device_data(),
             &zero,
             vector_y.device_data()));
    #endif
          cusparseSetStream(handle, 0);
          break;

      }
    }


    #endif



}

#undef HAS_CUSPARSE_GENERIC_API
#undef HAS_MKL_INSPECTOR_EXECUTOR_API

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_H
