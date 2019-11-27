#ifndef JAMS_CONTAINERS_SPARSE_MATRIX_BUILDER_H
#define JAMS_CONTAINERS_SPARSE_MATRIX_BUILDER_H

#include "jams/containers/sparse_matrix.h"

namespace jams {
    template<typename T>
    class SparseMatrix<T>::Builder {
    public:
        Builder() = default;

        Builder(size_type num_rows, size_type num_cols)
            : num_rows_(num_rows), num_cols_(num_cols) {}

        void insert(size_type i, size_type j, const value_type &value);

        std::size_t memory() {
          return val_.capacity() * sizeof(value_type)
               + row_.capacity() * sizeof(size_type)
               + col_.capacity() * sizeof(size_type);
        }

        SparseMatrix<T> build();

        inline SparseMatrixFormat format() const { return format_; }

        inline SparseMatrixType type() const { return type_; }

        inline SparseMatrixFillMode fill_mode() const { return fill_mode_; }

        inline SparseMatrix<T>::Builder &set_format(SparseMatrixFormat f) {
          format_ = f;
          return *this;
        }

        inline SparseMatrix<T>::Builder &set_type(SparseMatrixType t) {
          type_ = t;
          return *this;
        }

        inline SparseMatrix<T>::Builder &set_fill_mode(SparseMatrixFillMode m) {
          fill_mode_ = m;
          return *this;
        }

        void clear();

    private:
        SparseMatrix<T> build_csr();
        SparseMatrix<T> build_coo();

        void sort();

        void sum_duplicates();

        void assert_safe_numeric_limits() const;

        void assert_index_is_valid(size_type i, size_type j) const;

        SparseMatrixFormat format_ = SparseMatrixFormat::COO;
        SparseMatrixType type_ = SparseMatrixType::GENERAL;
        SparseMatrixFillMode fill_mode_ = SparseMatrixFillMode::LOWER;

        size_type num_rows_ = 0;
        size_type num_cols_ = 0;

        std::vector<value_type> val_;
        std::vector<size_type> row_;
        std::vector<size_type> col_;
    };

    template<typename T>
    void SparseMatrix<T>::Builder::insert(size_type i, size_type j, const value_type &value) {
      assert_safe_numeric_limits();
      assert_index_is_valid(i, j);
      row_.push_back(i);
      col_.push_back(j);
      val_.push_back(value);
    }

    template<typename T>
    void SparseMatrix<T>::Builder::sort() {
      auto p = stable_sort_permutation(col_);
      apply_permutation_in_place(row_, p);
      apply_permutation_in_place(col_, p);
      apply_permutation_in_place(val_, p);

      p = stable_sort_permutation(row_);
      apply_permutation_in_place(row_, p);
      apply_permutation_in_place(col_, p);
      apply_permutation_in_place(val_, p);
    }

    template<typename T>
    void SparseMatrix<T>::Builder::sum_duplicates() {
      for (auto m = 1; m < row_.size(); ++m) {
        if (row_[m] == row_[m - 1] && col_[m] == col_[m - 1]) {
          val_[m - 1] += val_[m];
          row_.erase(row_.begin() + m);
          col_.erase(col_.begin() + m);
        }
      }
    }

    template<typename T>
    void SparseMatrix<T>::Builder::clear() {
      jams::util::force_deallocation(row_);
      jams::util::force_deallocation(col_);
      jams::util::force_deallocation(val_);
    }

    template<typename T>
    SparseMatrix<T> SparseMatrix<T>::Builder::build() {
      switch (format_) {
        case SparseMatrixFormat::COO :
          return build_coo();
        case SparseMatrixFormat::CSR :
          return build_csr();
      };
    }

    template<typename T>
    SparseMatrix<T> SparseMatrix<T>::Builder::build_csr() {
      this->sort();
      this->sum_duplicates();

      const auto nnz = val_.size();

      index_container csr_rows(num_rows_ + 1);
      csr_rows(0) = 0;
      size_type current_row = 0;

      for (auto m = 1; m < nnz; ++m) {
        assert(m < row_.size());
        if (row_[m] == current_row) {
          continue;
        }
        assert(current_row + 1 < csr_rows.size());
        csr_rows(current_row + 1) = m;
        current_row++;
      }
      csr_rows(num_rows_) = nnz;
      jams::util::force_deallocation(row_);
      index_container csr_cols(col_.begin(), col_.end());
      jams::util::force_deallocation(col_);
      value_container csr_vals(val_.begin(), val_.end());
      jams::util::force_deallocation(val_);

      return SparseMatrix<T>(num_rows_, num_cols_, nnz, csr_rows, csr_cols, csr_vals, format_, type_, fill_mode_);
    }

    template<typename T>
    SparseMatrix<T> SparseMatrix<T>::Builder::build_coo() {
      this->sort();
      this->sum_duplicates();
      index_container coo_rows(row_.begin(), row_.end());
      jams::util::force_deallocation(row_);
      index_container coo_cols(col_.begin(), col_.end());
      jams::util::force_deallocation(col_);
      value_container coo_vals(val_.begin(), val_.end());
      jams::util::force_deallocation(val_);

      return SparseMatrix<T>(num_rows_, num_cols_, val_.size(), coo_rows, coo_cols, coo_vals, format_, type_, fill_mode_);
    }

    template<typename T>
    void SparseMatrix<T>::Builder::assert_safe_numeric_limits() const {
      if (val_.size() >= std::numeric_limits<size_type>::max() - 1) {
        throw std::runtime_error("Number of non zero elements is too large for the sparse matrix size_type");
      }
    }

    template<typename T>
    void SparseMatrix<T>::Builder::assert_index_is_valid(SparseMatrix::size_type i, SparseMatrix::size_type j) const {
      if ((i >= num_rows_) || (i < 0) || (j >= num_cols_) || (j < 0)) {
        throw std::runtime_error("Invalid index for sparse matrix");
      }
    }
};

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_BUILDER_H
