//
// Created by Joseph Barker on 2019-10-10.
//

#ifndef JAMS_CONTAINERS_SPARSE_MATRIX_DESCRIPTION_H
#define JAMS_CONTAINERS_SPARSE_MATRIX_DESCRIPTION_H

#include <array>

#ifdef HAS_CUDA
#include <cusparse.h>
#include "jams/cuda/cuda_common.h"
#endif

namespace jams {
    enum class SparseMatrixType {
        GENERAL, SYMMETRIC };

    enum class SparseMatrixFillMode {
        LOWER, UPPER };

    enum class SparseMatrixDiagType {
        NON_UNIT, UNIT };

    enum class SparseMatrixFormat {
        COO, CSR };

    // SparseMatrixDescription helps us to give information to different libraries about the sparse matrix.
    // The class is also movable which avoid issues with cusparseMatDescr_t (which is actually a pointer) in movable
    // containers.
    class SparseMatrixDescription {
    public:
        friend void swap(SparseMatrixDescription &lhs, SparseMatrixDescription &rhs);

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
        SparseMatrixDescription &operator=(SparseMatrixDescription rhs) &{
          swap(*this, rhs);
          return *this;
        }

        SparseMatrixDescription(const SparseMatrixDescription &rhs) :
            SparseMatrixDescription(rhs.format_, rhs.type_, rhs.fill_mode_, rhs.diag_type_) {}

        SparseMatrixDescription(SparseMatrixDescription &&rhs) noexcept :
            format_(rhs.format_), type_(rhs.type_), fill_mode_(rhs.fill_mode_), diag_type_(rhs.diag_type_),
            mkl_desc_(rhs.mkl_desc_) {
          #ifdef HAS_CUDA
          cusparse_desc_ = rhs.cusparse_desc_;
          rhs.cusparse_desc_ = nullptr;
          #endif
        };

        inline void set_type(SparseMatrixType type) {
          type_ = type;
          update_descriptors();
        }

        inline void set_format(SparseMatrixFormat format) {
          format_ = format;
          update_descriptors();
        }

        inline void set_fill_mode(SparseMatrixFillMode mode) {
          fill_mode_ = mode;
          update_descriptors();
        }

        inline void set_diag_type(SparseMatrixDiagType diag) {
          diag_type_ = diag;
          update_descriptors();
        }

        inline SparseMatrixType type() const { return type_; }

        inline SparseMatrixFormat format() const { return format_; }

        inline SparseMatrixFillMode fill_mode() const { return fill_mode_; }

        inline SparseMatrixDiagType diag_type() const { return diag_type_; }

        const char* mkl_desc() const { return mkl_desc_.data(); }

        #if HAS_CUDA
        cusparseMatDescr_t cusparse_desc() const { return cusparse_desc_; };
        #endif

    private:
        void update_descriptors() {
          set_mkl_descriptor();
          set_cusparse_descriptor();
        }

        void set_mkl_descriptor() {
          switch (type_) {
            case SparseMatrixType::GENERAL:
              mkl_desc_[0] = 'G';
              break;
            case SparseMatrixType::SYMMETRIC:
              mkl_desc_[0] = 'S';
              break;
          }

          switch (fill_mode_) {
            case SparseMatrixFillMode::LOWER:
              mkl_desc_[1] = 'L';
              break;
            case SparseMatrixFillMode::UPPER:
              mkl_desc_[1] = 'U';
              break;
          }

          switch (diag_type_) {
            case SparseMatrixDiagType::NON_UNIT:
              mkl_desc_[2] = 'N';
              break;
            case SparseMatrixDiagType::UNIT:
              mkl_desc_[2] = 'U';
              break;
          }

          mkl_desc_[3] = 'C'; // always use zero-based indexing
          mkl_desc_[4] = 'N'; // unused by MKL
          mkl_desc_[5] = 'N'; // unused by MKL
        }

        void set_cusparse_descriptor() {
          #if HAS_CUDA
          if (cusparse_desc_ == nullptr) {
            CHECK_CUSPARSE_STATUS(cusparseCreateMatDescr(&cusparse_desc_));
          }

          switch (type_) {
            case SparseMatrixType::GENERAL:
              cusparseSetMatType(cusparse_desc_, CUSPARSE_MATRIX_TYPE_GENERAL);
              break;
            case SparseMatrixType::SYMMETRIC:
              cusparseSetMatType(cusparse_desc_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
              break;
          }

          switch (fill_mode_) {
            case SparseMatrixFillMode::LOWER:
              cusparseSetMatFillMode(cusparse_desc_, CUSPARSE_FILL_MODE_LOWER);
              break;
            case SparseMatrixFillMode::UPPER:
              cusparseSetMatFillMode(cusparse_desc_, CUSPARSE_FILL_MODE_UPPER);
              break;
          }

          switch (diag_type_) {
            case SparseMatrixDiagType::NON_UNIT:
              cusparseSetMatDiagType(cusparse_desc_, CUSPARSE_DIAG_TYPE_NON_UNIT);
              break;
            case SparseMatrixDiagType::UNIT:
              cusparseSetMatDiagType(cusparse_desc_, CUSPARSE_DIAG_TYPE_UNIT);
              break;
          }
          #endif
        }

        // default values
        SparseMatrixType type_ = SparseMatrixType::GENERAL;
        SparseMatrixFormat format_ = SparseMatrixFormat::CSR;
        SparseMatrixFillMode fill_mode_ = SparseMatrixFillMode::LOWER;
        SparseMatrixDiagType diag_type_ = SparseMatrixDiagType::NON_UNIT;

        std::array<char, 6> mkl_desc_ = {'G', 'L', 'N', 'C', 'N', 'N'};

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
}


#endif //JAMS_CONTAINERS_SPARSE_MATRIX_DESCRIPTION_H
