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

#include "jams/containers/multiarray.h"
#include "jams/helpers/exception.h"
#include "jams/interface/openmp.h"
#include "jams/helpers/sort.h"

#if HAS_MKL
#include <mkl_spblas.h>
#endif

#ifdef HAS_CUDA
#include <cusparse.h>
#include "jams/cuda/cuda_common.h"
#endif


namespace jams {
    namespace impl {
        template<typename MatType, typename VecType>
        __attribute__((hot))
        VecType Xcsrmv_general_row(
            const MatType *csr_val,
            const int *csr_col,
            const int *csr_row,
            const VecType *x,
            const int row) {

          VecType sum = 0.0;
          for (auto j = csr_row[row]; j < csr_row[row + 1]; ++j) {
            sum += x[csr_col[j]] * csr_val[j];
          }
          return sum;
        }

        template<typename MatType, typename VecType>
        VecType Xcoomv_general_row(
            const MatType *coo_val,
            const int *coo_col,
            const int *coo_row,
            const VecType *x,
            const int row) {

          // coordinates must be sorted by row

          auto i = 0;
          // skip through until we hit the row of interest
          while (coo_row[i] < row) {
            ++i;
          }

          // process just the rows of interest and then finish
          VecType sum = 0.0;
          while(coo_row[i] == row) {
            auto col = coo_col[i];
            auto val = coo_val[i];
            sum += x[col] * val;
            ++i;
          }

          return sum;
        }

        template<typename MatType, typename VecType>
        __attribute__((hot))
        void Xcsrmv_general(
            const VecType &alpha,
            const VecType &beta,
            const int &m,
            const MatType *csr_val,
            const int *csr_col,
            const int *csr_row,
            const VecType *x,
            double *y) {

          if (alpha == 1.0 && beta == 0.0) {
            OMP_PARALLEL_FOR
            for (auto i = 0; i < m; ++i) {  // iterate num_rows
              y[i] = Xcsrmv_general_row(csr_val, csr_col, csr_row, x, i);
            }
          } else {
            OMP_PARALLEL_FOR
            for (auto i = 0; i < m; ++i) {  // iterate num_rows
              auto sum = Xcsrmv_general_row(csr_val, csr_col, csr_row, x, i);
              y[i] = beta * y[i] + alpha * sum;
            }
          }
        }

        template<typename MatType, typename VecType>
        __attribute__((hot))
        void Xcoomv_general(
            const VecType &alpha,
            const VecType &beta,
            const int &m,
            const int &nnz,
            const MatType *coo_val,
            const int *coo_col,
            const int *coo_row,
            const VecType *x,
            double *y) {

          if (alpha == 1.0 && beta == 0.0) {
            memset(y, 0.0, sizeof(double)*m);
            OMP_PARALLEL_FOR
            for (auto i = 0; i < nnz; ++i) {
              auto row = coo_row[i];
              auto col = coo_col[i];
              auto val = coo_val[i];
              y[row] += x[col] * val;
            }
          } else {
            OMP_PARALLEL_FOR
            for (auto i = 0; i < m; ++i) {
              y[i] *= beta;
            }
            OMP_PARALLEL_FOR
            for (auto i = 0; i < nnz; ++i) {
              auto row = coo_row[i];
              auto col = coo_col[i];
              auto val = coo_val[i];
              y[row] += alpha * x[col] * val;
            }
          }
        }

    }

    // enum integer values mirror those in mkl_spblas.h so we can static_cast if needed
    enum class SparseMatrixType {
        GENERAL = 20,
        SYMMETRIC = 21
    };

    enum class SparseMatrixFillMode {
        LOWER = 40,
        UPPER = 41
    };

    enum class SparseMatrixFormat {
        COO, CSR
    };


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

        template<class U, size_t N>
        void multiply(const MultiArray<U, N> &vector, MultiArray<U, N> &result) const;

        template<class U, size_t N>
        U multiply_row(const size_type i, const MultiArray<U, N> &vector) const;

        #if HAS_CUDA
        template <class U, size_t N>
        void multiply_gpu(const MultiArray<U, N> &vector, MultiArray<U, N> &result,
                          cusparseHandle_t &handle, cudaStream_t stream_id);
        #endif

        SparseMatrix() = default;

        SparseMatrix(size_type num_rows, size_type num_cols, size_type num_non_zero,
            index_container &rows, index_container &cols, value_container &vals,
            SparseMatrixFormat format, SparseMatrixType type, SparseMatrixFillMode fill_mode)
            : num_rows_(num_rows),
              num_cols_(num_cols),
              num_non_zero_(num_non_zero),
              row_(rows),
              col_(cols),
              val_(vals),
              format_(format),
              type_(type),
              fill_mode_(fill_mode) {
#ifdef HAS_CUDA
          cusparseCreateMatDescr(&cusparse_descr_);
          switch (type_) {
            case SparseMatrixType::GENERAL:
              cusparseSetMatType(cusparse_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
              break;
            case SparseMatrixType::SYMMETRIC:
              cusparseSetMatType(cusparse_descr_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
              break;
          }

          switch (fill_mode_) {
            case SparseMatrixFillMode::LOWER:
              cusparseSetMatFillMode(cusparse_descr_, CUSPARSE_FILL_MODE_LOWER);
              break;
            case SparseMatrixFillMode::UPPER:
              cusparseSetMatFillMode(cusparse_descr_, CUSPARSE_FILL_MODE_UPPER);
              break;
          }
#endif
        }

        inline SparseMatrixFormat format() const { return format_; }

        inline SparseMatrixType type() const { return type_; }

        inline SparseMatrixFillMode fill_mode() const { return fill_mode_; }

        inline size_type num_non_zero() const { return num_non_zero_; }

        inline size_type num_rows() const { return num_rows_; }

        inline size_type num_cols() const { return num_cols_; }

        std::size_t memory() { return row_.memory() + col_.memory() + val_.memory(); };

    protected:
        SparseMatrixFormat format_ = SparseMatrixFormat::COO;

        SparseMatrixType type_ = SparseMatrixType::GENERAL;
        SparseMatrixFillMode fill_mode_ = SparseMatrixFillMode::LOWER;

        size_type num_rows_ = 0;
        size_type num_cols_ = 0;
        size_type num_non_zero_ = 0;

        index_container row_;
        index_container col_;
        value_container val_;

        #ifdef HAS_CUDA
        template <class U, size_t N>
        void allocate_cusparse_csr_buffer(const MultiArray<U, N> &vector, MultiArray<U, N> &result, cusparseHandle_t &handle);

        cusparseMatDescr_t cusparse_descr_ = nullptr;

        MultiArray<char, 1> cusparse_buffer_;
        #endif
    };


    template<typename T>
    template<class U, size_t N>
    void SparseMatrix<T>::multiply(const MultiArray<U, N> &vector, MultiArray<U, N> &result) const {
      switch (format_) {
        case SparseMatrixFormat::COO:
          impl::Xcoomv_general(1.0, 0.0, num_rows_, num_non_zero_,
                               val_.data(), col_.data(), row_.data(),
                               vector.data(), result.data());
//          throw std::runtime_error("unimplemented");
        case SparseMatrixFormat::CSR:
          #if HAS_MKL
          double one = 1.0;
          double zero = 0.0;
          const char transa[1] = {'N'};
          const char matdescra[6] = {'G', 'L', 'N', 'C', 'N', 'N'};
          mkl_dcsrmv(transa, &num_rows_, &num_cols_, &one, matdescra, val_.data(),
                     col_.data(), row_.data(), row_.data() + 1, vector.data(),
                     &zero, result.data());
          #else
          impl::Xcsrmv_general(
              1.0, 0.0, num_rows_,
              val_.data(), col_.data(), row_.data(),
              vector.data(), result.data());
          #endif
      }
    }

    template<typename T>
    template<class U, size_t N>
    U SparseMatrix<T>::multiply_row(const size_type i, const MultiArray<U, N> &vector) const {
      switch (format_) {
        case SparseMatrixFormat::COO:
          return impl::Xcoomv_general_row(
              val_.data(), col_.data(), row_.data(),
              vector.data(), i);
        case SparseMatrixFormat::CSR:
          return impl::Xcsrmv_general_row(
              val_.data(), col_.data(), row_.data(),
              vector.data(), i);
      }
    }

    template<typename T>
    template<class U, size_t N>
    void SparseMatrix<T>::allocate_cusparse_csr_buffer(const MultiArray<U, N> &vector, MultiArray<U, N> &result, cusparseHandle_t &handle) {
      if (cusparse_buffer_.empty()) {
        const T one = 1.0;
        const T zero = 0.0;
        size_t buffer_size = 0;
        CHECK_CUSPARSE_STATUS(cusparseCsrmvEx_bufferSize(
            handle,
            CUSPARSE_ALG0, //default, naive
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            num_rows_,
            num_cols_,
            num_non_zero_,
            &one, gpu::get_cuda_data_type<T>(),
            cusparse_descr_,
            val_.device_data(), gpu::get_cuda_data_type<T>(),
            row_.device_data(),
            col_.device_data(),
            vector.device_data(), gpu::get_cuda_data_type<U>(), // this is presumably not used?
            &zero, gpu::get_cuda_data_type<T>(),
            result.device_data(), gpu::get_cuda_data_type<U>(), // this is presumably not used?
            gpu::get_cuda_data_type<double>(), // execution type
            &buffer_size));
        cusparse_buffer_.resize(buffer_size);
      }
    }

    template<typename T>
    template<class U, size_t N>
    void SparseMatrix<T>::multiply_gpu(const MultiArray<U, N> &vector, MultiArray<U, N> &result,
                                       cusparseHandle_t &handle, cudaStream_t stream_id) {
      assert(handle != nullptr);
      assert(cusparse_descr_ != nullptr);
      if (cusparse_buffer_.empty()) {
        allocate_cusparse_csr_buffer(vector, result, handle);
      }
      assert(cusparse_buffer_.device_data() != nullptr);
      const T one = 1.0;
      const T zero = 0.0;
      switch (format_) {
        case SparseMatrixFormat::COO:
          throw std::runtime_error("unimplemented");
        case SparseMatrixFormat::CSR:
          cusparseSetStream(handle, stream_id);
          CHECK_CUSPARSE_STATUS(cusparseCsrmvEx(
              handle,
              CUSPARSE_ALG0,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              num_rows_,
              num_cols_,
              num_non_zero_,
              &one, gpu::get_cuda_data_type<T>(),
              cusparse_descr_,
              val_.device_data(), gpu::get_cuda_data_type<T>(),
              row_.device_data(),
              col_.device_data(),
              vector.device_data(), gpu::get_cuda_data_type<double>(), // this is presumably not used?
              &zero, gpu::get_cuda_data_type<T>(),
              result.device_data(), gpu::get_cuda_data_type<double>(), // this is presumably not used?
              gpu::get_cuda_data_type<double>(), // execution type
              (void*)(cusparse_buffer_.device_data())));
          cusparseSetStream(handle, 0);

      }
    }
}



//
//      if(type_ == SparseMatrixType::SYMMETRIC) {
//        if(fill_mode_ == SparseMatrixFillMode::LOWER) {
//          if( i > j ) {
//            throw std::runtime_error("Attempted to insert lower matrix element in symmetric upper sparse matrix");
//          }
//        } else {
//          if( i < j ) {
//            throw std::runtime_error("Attempted to insert upper matrix element in symmetric lower sparse matrix");
//          }
//        }
//      }
//    }




#endif // JAMS_CONTAINERS_SPARSE_MATRIX_H
