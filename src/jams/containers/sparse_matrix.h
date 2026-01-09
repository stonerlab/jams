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
#include <type_traits>

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

        SparseMatrix(SparseMatrix&& rhs) noexcept
        : matrix_A_description_(std::move(rhs.matrix_A_description_)),
          num_rows_(std::move(rhs.num_rows_)),
          num_cols_(std::move(rhs.num_cols_)),
          num_non_zero_(std::move(rhs.num_non_zero_)),
          row_(std::move(rhs.row_)),
          col_(std::move(rhs.col_)),
          val_(std::move(rhs.val_)) {

          #if HAS_MKL_INSPECTOR_EXECUTOR_API
          mkl_matrix_A_handle_ = rhs.mkl_matrix_A_handle_;
          rhs.mkl_matrix_A_handle_ = nullptr;
          #endif

          #if HAS_CUSPARSE_GENERIC_API
          cusparse_matrix_A_handle_ = rhs.cusparse_matrix_A_handle_;
          rhs.cusparse_matrix_A_handle_ = nullptr;

          cusparse_vector_x_handle_ = rhs.cusparse_vector_x_handle_;
          rhs.cusparse_vector_x_handle_ = nullptr;

          cusparse_vector_y_handle_ = rhs.cusparse_vector_y_handle_;
          rhs.cusparse_vector_y_handle_ = nullptr;

          cusparse_buffer_ = rhs.cusparse_buffer_;
          rhs.cusparse_buffer_ = nullptr;

          cusparse_buffer_size_ = rhs.cusparse_buffer_size_;
          rhs.cusparse_buffer_size_ = 0;

          cusparse_cached_x_elements_ = rhs.cusparse_cached_x_elements_;
          rhs.cusparse_cached_x_elements_ = 0;

          cusparse_cached_y_elements_ = rhs.cusparse_cached_y_elements_;
          rhs.cusparse_cached_y_elements_ = 0;

          cusparse_cached_compute_type_ = rhs.cusparse_cached_compute_type_;
          rhs.cusparse_cached_compute_type_ = CUDA_R_32F;

          cusparse_cached_alg_ = rhs.cusparse_cached_alg_;
          rhs.cusparse_cached_alg_ = CUSPARSE_SPMV_ALG_DEFAULT;

          cusparse_buffer_valid_ = rhs.cusparse_buffer_valid_;
          rhs.cusparse_buffer_valid_ = false;
          #endif
        }

//        // copy assign
//        SparseMatrix &operator=(SparseMatrix rhs) &{
//          swap(*this, rhs);
//          return *this;
//        }

        SparseMatrix &operator=(SparseMatrix&& rhs) noexcept {
          if (this != &rhs) {
            // Release any backend resources currently owned by *this* before stealing from rhs.
            release_backend_handles_();
            matrix_A_description_ = rhs.matrix_A_description_;
            num_rows_ = rhs.num_rows_;
            num_cols_ = rhs.num_cols_;
            num_non_zero_ = rhs.num_non_zero_;
            row_ = std::move(rhs.row_);
            col_ = std::move(rhs.col_);
            val_ = std::move(rhs.val_);

            #if HAS_MKL_INSPECTOR_EXECUTOR_API
            mkl_matrix_A_handle_ = rhs.mkl_matrix_A_handle_;
            rhs.mkl_matrix_A_handle_ = nullptr;
            #endif

            #if HAS_CUSPARSE_GENERIC_API
            cusparse_matrix_A_handle_ = rhs.cusparse_matrix_A_handle_;
            rhs.cusparse_matrix_A_handle_ = nullptr;

            cusparse_vector_x_handle_ = rhs.cusparse_vector_x_handle_;
            rhs.cusparse_vector_x_handle_ = nullptr;

            cusparse_vector_y_handle_ = rhs.cusparse_vector_y_handle_;
            rhs.cusparse_vector_y_handle_ = nullptr;

            cusparse_buffer_ = rhs.cusparse_buffer_;
            rhs.cusparse_buffer_ = nullptr;

            cusparse_buffer_size_ = rhs.cusparse_buffer_size_;
            rhs.cusparse_buffer_size_ = 0;

            cusparse_cached_x_elements_ = rhs.cusparse_cached_x_elements_;
            rhs.cusparse_cached_x_elements_ = 0;

            cusparse_cached_y_elements_ = rhs.cusparse_cached_y_elements_;
            rhs.cusparse_cached_y_elements_ = 0;

            cusparse_cached_compute_type_ = rhs.cusparse_cached_compute_type_;
            rhs.cusparse_cached_compute_type_ = CUDA_R_32F;

            cusparse_cached_alg_ = rhs.cusparse_cached_alg_;
            rhs.cusparse_cached_alg_ = CUSPARSE_SPMV_ALG_DEFAULT;

            cusparse_buffer_valid_ = rhs.cusparse_buffer_valid_;
            rhs.cusparse_buffer_valid_ = false;
            #endif
          }
          return *this;
        }

        inline ~SparseMatrix() {
          release_backend_handles_();
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

        inline constexpr std::size_t memory() const { return row_.bytes() + col_.bytes() + val_.bytes(); };

        // Performs the multiplication y = A * x where 'A' is this sparse matrix
        // and vectors x and y are dense vectors passed into the function.
        template<class X, class Y, size_t N>
        void multiply(const MultiArray<X, N> &vector_x, MultiArray<Y, N> &vector_y);


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
        // Cached SpMV configuration so we can avoid calling cusparseSpMV_bufferSize on every multiply.
        std::size_t cusparse_cached_x_elements_ = 0;
        std::size_t cusparse_cached_y_elements_ = 0;
        cudaDataType cusparse_cached_compute_type_ = CUDA_R_32F; // sentinel
        cusparseSpMVAlg_t cusparse_cached_alg_ = CUSPARSE_SPMV_ALG_DEFAULT;
        bool cusparse_buffer_valid_ = false;
        #endif

        index_type num_rows_             = 0;
        index_type num_cols_             = 0;
        index_type num_non_zero_         = 0;
        index_container row_;
        index_container col_;
        value_container val_;

        void release_backend_handles_() noexcept {
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

          if (cusparse_vector_y_handle_) {
            cusparseDestroyDnVec(cusparse_vector_y_handle_);
            cusparse_vector_y_handle_ = nullptr;
          }

          if (cusparse_buffer_) {
            cudaFree(cusparse_buffer_);
            cusparse_buffer_ = nullptr;
          }
          cusparse_buffer_size_ = 0;
          cusparse_cached_x_elements_ = 0;
          cusparse_cached_y_elements_ = 0;
          cusparse_cached_compute_type_ = CUDA_R_32F; // sentinel
          cusparse_cached_alg_ = CUSPARSE_SPMV_ALG_DEFAULT;
          cusparse_buffer_valid_ = false;
          #endif
        }
    };

template<typename T>
template<class X, class Y, size_t N>
void SparseMatrix<T>::multiply(const MultiArray<X, N> &vector_x, MultiArray<Y, N> &vector_y) {
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

        // MKL Inspector Executor disabled on 2024-01-10 due to segfaulting.
        //            #if HAS_MKL_INSPECTOR_EXECUTOR_API
        //              if (!mkl_matrix_A_handle_) {
        //                sparse_status_t status = mkl_sparse_d_create_csr(&mkl_matrix_A_handle_,
        //                                        SPARSE_INDEX_BASE_ZERO,
        //                                        num_rows_, num_cols_,
        //                                        row_.data(), row_.data() + 1,
        //                                        col_.data(), val_.data());
        //                assert(status == SPARSE_STATUS_SUCCESS);
        //              }
        //              mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
        //                              1.0, mkl_matrix_A_handle_,
        //                              matrix_A_description_.mkl_inspector_executor_desc(),
        //                              vector_x.data(), 0.0, vector_y.data());
        //            #else

        // Only use MKL when X and Y have the same type and the call is MKL-compatible.
        // If X and Y differ, we must use the jams::* kernels.
        if constexpr (std::is_same<X, Y>::value &&
                      std::is_same<T, double>::value &&
                      std::is_same<X, double>::value) {
          double one = 1.0, zero = 0.0;
          const char transa[1] = {'N'};
          mkl_dcsrmv(transa, &num_rows_, &num_cols_, &one, matrix_A_description_.mkl_desc(),
                     val_.data(), col_.data(), row_.data(), row_.data() + 1,
                     vector_x.data(), &zero, vector_y.data());
        } else {
          jams::Xcsrmv_general(
              1.0, 0.0, num_rows_, val_.data(), col_.data(), row_.data(), vector_x.data(), vector_y.data());
        }

        //            #endif
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
          // Preserve the handle's existing stream and restore it on exit.
          cudaStream_t prev_stream = nullptr;
          CHECK_CUSPARSE_STATUS(cusparseGetStream(handle, &prev_stream));
          CHECK_CUSPARSE_STATUS(cusparseSetStream(handle, stream_id));
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

          if (cusparse_vector_x_handle_ && cusparse_cached_x_elements_ != vector_x.elements()) {
            CHECK_CUSPARSE_STATUS(cusparseDestroyDnVec(cusparse_vector_x_handle_));
            cusparse_vector_x_handle_ = nullptr;
            cusparse_buffer_valid_ = false;
          }
          if (!cusparse_vector_x_handle_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateDnVec(
                &cusparse_vector_x_handle_,
                vector_x.elements(),
                (void*)vector_x.device_data(),
                cuda::get_cuda_data_type<U>()));
          }
          CHECK_CUSPARSE_STATUS(cusparseDnVecSetValues(cusparse_vector_x_handle_, (void*)vector_x.device_data()));

          if (cusparse_vector_y_handle_ && cusparse_cached_y_elements_ != vector_y.elements()) {
            CHECK_CUSPARSE_STATUS(cusparseDestroyDnVec(cusparse_vector_y_handle_));
            cusparse_vector_y_handle_ = nullptr;
            cusparse_buffer_valid_ = false;
          }
          if (!cusparse_vector_y_handle_) {
            CHECK_CUSPARSE_STATUS(cusparseCreateDnVec(
                &cusparse_vector_y_handle_,
                vector_y.elements(),
                (void*)vector_y.device_data(),
                cuda::get_cuda_data_type<U>()));
          }
          CHECK_CUSPARSE_STATUS(cusparseDnVecSetValues(cusparse_vector_y_handle_, (void*)vector_y.device_data()));

          const cudaDataType compute_type = cuda::get_cuda_data_type<U>();
          const cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

          const bool need_buffer_query = (!cusparse_buffer_valid_) ||
                                         (cusparse_cached_x_elements_ != vector_x.elements()) ||
                                         (cusparse_cached_y_elements_ != vector_y.elements()) ||
                                         (cusparse_cached_compute_type_ != compute_type) ||
                                         (cusparse_cached_alg_ != alg);

          if (need_buffer_query) {
            size_t new_buffer_size = 0;
            CHECK_CUSPARSE_STATUS(cusparseSpMV_bufferSize(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                cusparse_matrix_A_handle_,
                cusparse_vector_x_handle_,
                &zero,
                cusparse_vector_y_handle_,
                compute_type,
                alg,
                &new_buffer_size));

            if (new_buffer_size > cusparse_buffer_size_) {
              if (cusparse_buffer_) {
                cudaFree(cusparse_buffer_);
              }
              cudaMalloc(&cusparse_buffer_, new_buffer_size);
              cusparse_buffer_size_ = new_buffer_size;
            }

            cusparse_cached_x_elements_ = vector_x.elements();
            cusparse_cached_y_elements_ = vector_y.elements();
            cusparse_cached_compute_type_ = compute_type;
            cusparse_cached_alg_ = alg;
            cusparse_buffer_valid_ = true;

            // Preprocess is tied to the current SpMV configuration and buffer.
            CHECK_CUSPARSE_STATUS(cusparseSpMV_preprocess(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                cusparse_matrix_A_handle_,
                cusparse_vector_x_handle_,
                &zero,
                cusparse_vector_y_handle_,
                compute_type,
                alg,
                cusparse_buffer_));
          }

          CHECK_CUSPARSE_STATUS(cusparseSpMV(
              handle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              &one,
              cusparse_matrix_A_handle_,
              cusparse_vector_x_handle_,
              &zero,
              cusparse_vector_y_handle_,
              compute_type,
              alg,
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
          // Restore the previous stream on the handle.
          CHECK_CUSPARSE_STATUS(cusparseSetStream(handle, prev_stream));
          break;

      }
    }


    #endif



}

#undef HAS_CUSPARSE_GENERIC_API
#undef HAS_MKL_INSPECTOR_EXECUTOR_API

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_H
