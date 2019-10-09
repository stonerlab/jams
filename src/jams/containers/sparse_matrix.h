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

    enum class SparseMatrixType { GENERAL, SYMMETRIC };

    enum class SparseMatrixFillMode { LOWER, UPPER };

    enum class SparseMatrixDiagType { NON_UNIT, UNIT };

    enum class SparseMatrixFormat { COO, CSR };

    // this class helps us to give information to different libraries about the sparse matrix

    class SparseMatrixDescription {
    public:
        friend void swap(SparseMatrixDescription& lhs, SparseMatrixDescription& rhs);

        SparseMatrixDescription() {
          update_descriptors();
        }

        SparseMatrixDescription(SparseMatrixFormat format, SparseMatrixType type,
            SparseMatrixFillMode fill_mode, SparseMatrixDiagType diag_type) :
            format_(format), type_(type), fill_mode_(fill_mode), diag_type_(diag_type) {
          update_descriptors();
        }

        ~SparseMatrixDescription() {
          #ifdef HAS_CUDA
          if (cusparse_desc_) {
            cusparseDestroyMatDescr(cusparse_desc_);
          }
          #endif
        };

        // copy assign
        SparseMatrixDescription& operator=(SparseMatrixDescription rhs) & {
          swap(*this, rhs);
          return *this;
        }

        SparseMatrixDescription(const SparseMatrixDescription& rhs) :
          SparseMatrixDescription(rhs.format_, rhs.type_, rhs.fill_mode_, rhs.diag_type_) {}

        SparseMatrixDescription(SparseMatrixDescription&& rhs) noexcept :
          format_(rhs.format_), type_(rhs.type_), fill_mode_(rhs.fill_mode_), diag_type_(rhs.diag_type_), mkl_desc_(rhs.mkl_desc_) {
          #ifdef HAS_CUDA
          cusparse_desc_ = rhs.cusparse_desc_;
          rhs.cusparse_desc_ = nullptr;
          #endif
        };

        inline void set_type(SparseMatrixType type) { type_ = type; update_descriptors(); }
        inline void set_format(SparseMatrixFormat format) { format_ = format; update_descriptors(); }
        inline void set_fill_mode(SparseMatrixFillMode mode ) { fill_mode_ = mode; update_descriptors(); }
        inline void set_diag_type(SparseMatrixDiagType diag ) { diag_type_ = diag; update_descriptors(); }

        inline SparseMatrixType type() const { return type_; }
        inline SparseMatrixFormat format() const { return format_; }
        inline SparseMatrixFillMode fill_mode() const { return fill_mode_; }
        inline SparseMatrixDiagType diag_type() const { return diag_type_; }

        const char * mkl_desc() const { return mkl_desc_.data(); }
        cusparseMatDescr_t cusparse_desc() { return cusparse_desc_; };

    private:
        void update_descriptors() {
          set_mkl_descriptor();
          set_cusparse_descriptor();
        }

        void set_mkl_descriptor() {
          switch (type_) {
            case SparseMatrixType::GENERAL:
              mkl_desc_[0] = 'G';
            case SparseMatrixType::SYMMETRIC:
              mkl_desc_[0] = 'S';
          }

          switch (fill_mode_) {
            case SparseMatrixFillMode::LOWER:
              mkl_desc_[1] = 'L';
            case SparseMatrixFillMode::UPPER:
              mkl_desc_[1] = 'U';
          }

          switch (diag_type_) {
            case SparseMatrixDiagType::NON_UNIT:
              mkl_desc_[2] = 'N';
            case SparseMatrixDiagType::UNIT:
              mkl_desc_[2] = 'U';
          }

          mkl_desc_[3] = 'C'; // always use zero-based indexing
          mkl_desc_[4] = 'N'; // unused by MKL
          mkl_desc_[5] = 'N'; // unused by MKL
        }

        void set_cusparse_descriptor() {
          if (cusparse_desc_ == nullptr) {
            CHECK_CUSPARSE_STATUS(cusparseCreateMatDescr(&cusparse_desc_));
          }

          switch (type_) {
            case SparseMatrixType::GENERAL:
              cusparseSetMatType(cusparse_desc_, CUSPARSE_MATRIX_TYPE_GENERAL);
            case SparseMatrixType::SYMMETRIC:
              cusparseSetMatType(cusparse_desc_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
          }

          switch (fill_mode_) {
            case SparseMatrixFillMode::LOWER:
              cusparseSetMatFillMode(cusparse_desc_, CUSPARSE_FILL_MODE_LOWER);
            case SparseMatrixFillMode::UPPER:
              cusparseSetMatFillMode(cusparse_desc_, CUSPARSE_FILL_MODE_UPPER);
          }

          switch (diag_type_) {
            case SparseMatrixDiagType::NON_UNIT:
              cusparseSetMatDiagType(cusparse_desc_, CUSPARSE_DIAG_TYPE_NON_UNIT);
            case SparseMatrixDiagType::UNIT:
              cusparseSetMatDiagType(cusparse_desc_, CUSPARSE_DIAG_TYPE_UNIT);
          }
        }

        // default values
        SparseMatrixType type_          = SparseMatrixType::GENERAL;
        SparseMatrixFormat format_      = SparseMatrixFormat::CSR;
        SparseMatrixFillMode fill_mode_ = SparseMatrixFillMode::LOWER;
        SparseMatrixDiagType diag_type_ = SparseMatrixDiagType::NON_UNIT;

        std::array<char,6> mkl_desc_ = {'G', 'L', 'N', 'C', 'N', 'N'};

        #if HAS_CUDA
        cusparseMatDescr_t cusparse_desc_ = nullptr;
        #endif
    };

    inline void swap(SparseMatrixDescription &lhs, SparseMatrixDescription &rhs) {
      using std::swap;
      swap(lhs.format_, rhs.format_);
      swap(lhs.type_, rhs.type_);
      swap(lhs.fill_mode_, rhs.fill_mode_);
      swap(lhs.diag_type_, rhs.diag_type_);
      swap(lhs.mkl_desc_, rhs.mkl_desc_);
      #ifdef HAS_CUDA
      swap(lhs.cusparse_desc_, rhs.cusparse_desc_);
      #endif
    }


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
          impl::Xcoomv_general(1.0, 0.0, num_rows_, num_non_zero_,
                               val_.data(), col_.data(), row_.data(),
                               vector.data(), result.data());
        case SparseMatrixFormat::CSR:
          #if HAS_MKL
          double one = 1.0;
          double zero = 0.0;
          const char transa[1] = {'N'};
          mkl_dcsrmv(transa, &num_rows_, &num_cols_, &one, description_.mkl_desc(), val_.data(),
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
      switch (description_.format()) {
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
      const T one = 1.0;
      const T zero = 0.0;
      switch (description_.format()) {
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
             description_.cusparse_desc(),
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

}

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_H
