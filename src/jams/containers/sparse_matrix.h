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

//        #ifdef HAS_CUDA
//        // create a class for this so we can use default copy and move constructors in SparseMatrix
//        class CusparseDescription {
//            public:
//            CusparseDescription() {
//              std::cout << "default construct" << std::endl;
//              CHECK_CUSPARSE_STATUS(cusparseCreateMatDescr(&descr_));
//            }
//
//            CusparseDescription(const CusparseDescription& rhs) {
//              std::cout << "copy construct" << std::endl;
//
//              CHECK_CUSPARSE_STATUS(cusparseCreateMatDescr(&descr_));
//              set_type(rhs.type());
//              set_fill_mode(rhs.fill_mode());
//            }
//
//            cusparseMatDescr_t get() const {
//              assert(descr_ != nullptr);
//              return descr_;
//            }
//
//            void set_type(cusparseMatrixType_t type) {
//              CHECK_CUSPARSE_STATUS(cusparseSetMatType(descr_, type));
//            }
//
//            void set_fill_mode(cusparseFillMode_t mode) {
//              CHECK_CUSPARSE_STATUS(cusparseSetMatFillMode(descr_, mode));
//            }
//
//            cusparseMatrixType_t type() const {
//              return cusparseGetMatType(descr_);
//            }
//
//            cusparseFillMode_t fill_mode() const {
//              return cusparseGetMatFillMode(descr_);
//            }
//
//            ~CusparseDescription() {
//              if (descr_){
//                std::cout << "destruct" << std::endl;
//                cusparseDestroyMatDescr(descr_);
//              }
//            }
//            private:
//
//            cusparseMatDescr_t descr_ = nullptr;
//
//            };
//        #endif

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

        template<class F>
        friend void swap(SparseMatrix<F>& lhs, SparseMatrix<F>& rhs);

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

        ~SparseMatrix() {
          #ifdef HAS_CUDA
          if (cusparse_descr_) {
            cusparseDestroyMatDescr(cusparse_descr_);
          }
          #endif
        }

        SparseMatrix(const size_type num_rows, const size_type num_cols, const size_type num_non_zero,
                     index_container rows, index_container cols, value_container vals,
                     SparseMatrixFormat format, SparseMatrixType type, SparseMatrixFillMode fill_mode)
            : format_(format),
              type_(type),
              fill_mode_(fill_mode),
              num_rows_(num_rows),
              num_cols_(num_cols),
              num_non_zero_(num_non_zero),
              row_(std::move(rows)),
              col_(std::move(cols)),
              val_(std::move(vals)) {
#ifdef HAS_CUDA
          cusparseCreateMatDescr(&cusparse_descr_);
          switch (type_) {
            case SparseMatrixType::GENERAL:
              cusparseSetMatType(cusparse_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
//            cusparse_descr_.set_type(CUSPARSE_MATRIX_TYPE_GENERAL);
              break;
            case SparseMatrixType::SYMMETRIC:
              cusparseSetMatType(cusparse_descr_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
//              cusparse_descr_.set_type(CUSPARSE_MATRIX_TYPE_SYMMETRIC);
//              switch (fill_mode_) {
//                case SparseMatrixFillMode::LOWER:
//                  cusparse_descr_.set_fill_mode(CUSPARSE_FILL_MODE_LOWER);
//                  break;
//                case SparseMatrixFillMode::UPPER:
//                  cusparse_descr_.set_fill_mode(CUSPARSE_FILL_MODE_UPPER);
//                  break;
//              }
              break;
          }
#endif
        }

        SparseMatrix(const SparseMatrix<T>& rhs) {
          SparseMatrix(rhs.num_rows_, rhs.num_cols_, rhs.num_non_zero_, rhs.row_, rhs.col_, rhs.val_, rhs.format_,
                       rhs.type_, rhs.fill_mode_);
        }

        SparseMatrix(SparseMatrix&& rhs) noexcept :
            format_(std::move(rhs.format_)),
            type_(std::move(rhs.type_)),
            fill_mode_(std::move(rhs.fill_mode_)),
            num_rows_(std::move(rhs.num_rows_)),
            num_cols_(std::move(rhs.num_cols_)),
            num_non_zero_(std::move(rhs.num_non_zero_)),
            row_(std::move(rhs.row_)),
            col_(std::move(rhs.col_)),
            val_(std::move(rhs.val_))
        {
          #ifdef HAS_CUDA
          cusparse_descr_ = std::move(rhs.cusparse_descr_);
          rhs.cusparse_descr_ = nullptr;
          #endif
        };

        // copy assign
        SparseMatrix& operator=(SparseMatrix rhs) & {
          swap(*this, rhs);
          return *this;
        }

        inline constexpr SparseMatrixFormat format() const { return format_; }

        inline constexpr SparseMatrixType type() const { return type_; }

        inline constexpr SparseMatrixFillMode fill_mode() const { return fill_mode_; }

        inline constexpr size_type num_non_zero() const { return num_non_zero_; }

        inline constexpr size_type num_rows() const { return num_rows_; }

        inline constexpr size_type num_cols() const { return num_cols_; }

        inline constexpr std::size_t memory() const { return row_.memory() + col_.memory() + val_.memory(); };

        template<class U, size_t N>
        void multiply(const MultiArray<U, N> &vector, MultiArray<U, N> &result) const;

        template<class U, size_t N>
        U multiply_row(size_type i, const MultiArray<U, N> &vector) const;

        #if HAS_CUDA
        template <class U, size_t N>
        void multiply_gpu(const MultiArray<U, N> &vector, MultiArray<U, N> &result,
                          cusparseHandle_t &handle, cudaStream_t stream_id);
        #endif

    protected:
        SparseMatrixFormat format_      = SparseMatrixFormat::CSR;
        SparseMatrixType type_          = SparseMatrixType::GENERAL;
        SparseMatrixFillMode fill_mode_ = SparseMatrixFillMode::LOWER;
        size_type num_rows_             = 0;
        size_type num_cols_             = 0;
        size_type num_non_zero_         = 0;
        index_container row_;
        index_container col_;
        value_container val_;

        #ifdef HAS_CUDA
        cusparseMatDescr_t cusparse_descr_ = nullptr;
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

    #ifdef HAS_CUDA
    template<typename T>
    template<class U, size_t N>
    void SparseMatrix<T>::multiply_gpu(const MultiArray<U, N> &vector, MultiArray<U, N> &result,
                                       cusparseHandle_t &handle, cudaStream_t stream_id) {
      assert(handle != nullptr);
      assert(cusparse_descr_ != nullptr);
      const T one = 1.0;
      const T zero = 0.0;
      switch (format_) {
        case SparseMatrixFormat::COO:
          throw std::runtime_error("unimplemented");
        case SparseMatrixFormat::CSR:
          cusparseSetStream(handle, stream_id);
          // TODO: implement CUDA 10 general type version
          // remember to (static)assert types are all the same if CUDA < 10

          CHECK_CUSPARSE_STATUS(cusparseDcsrmv(
             handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             num_rows_,
             num_cols_,
             num_non_zero_,
             &one,
             cusparse_descr_,
             val_.device_data(),
             row_.device_data(),
             col_.device_data(),
             vector.device_data(),
             &zero,
             result.device_data()));
          cusparseSetStream(handle, 0);
      }
    }
    #endif

    template<class F>
    void swap(SparseMatrix<F> &lhs, SparseMatrix<F> &rhs) {
      using std::swap;
      swap(lhs.format_, rhs.format_);
      swap(lhs.type_, rhs.type_);
      swap(lhs.fill_mode_, rhs.fill_mode_);
      swap(lhs.num_rows_, rhs.num_rows_);
      swap(lhs.num_cols_, rhs.num_cols_);
      swap(lhs.num_non_zero_, rhs.num_non_zero_);
      swap(lhs.row_, rhs.row_);
      swap(lhs.col_, rhs.col_);
      swap(lhs.val_, rhs.val_);
      #ifdef HAS_CUDA
      swap(lhs.cusparse_descr_, rhs.cusparse_descr_);
      #endif
    }

}

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_H
