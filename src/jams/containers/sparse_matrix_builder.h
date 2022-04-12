#ifndef JAMS_CONTAINERS_SPARSE_MATRIX_BUILDER_H
#define JAMS_CONTAINERS_SPARSE_MATRIX_BUILDER_H

#include <iostream>

#include "jams/containers/sparse_matrix.h"

namespace jams {

    enum class SparseMatrixSymmetryCheck {
        None,
        Symmetric,
        StructurallySymmetric
    };

    template<typename T>
    class SparseMatrix<T>::Builder {
    public:
        Builder() = default;

        Builder(index_type num_rows, index_type num_cols)
            : num_rows_(num_rows), num_cols_(num_cols) {}

        void insert(index_type i, index_type j, const value_type &value);

        std::size_t memory() {
          return val_.capacity() * sizeof(value_type)
               + row_.capacity() * sizeof(index_type)
               + col_.capacity() * sizeof(index_type);
        }

        void output(std::ostream& os);

        bool is_structurally_symmetric();
        bool is_symmetric();

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
        void merge();

        void assert_safe_numeric_limits() const;

        void assert_index_is_valid(index_type i, index_type j) const;

        SparseMatrixFormat format_ = SparseMatrixFormat::COO;
        SparseMatrixType type_ = SparseMatrixType::GENERAL;
        SparseMatrixFillMode fill_mode_ = SparseMatrixFillMode::LOWER;

        index_type num_rows_ = 0;
        index_type num_cols_ = 0;

        bool is_sorted_ = false;
        bool is_merged_ = false;

        std::vector<value_type> val_;
        std::vector<index_type> row_;
        std::vector<index_type> col_;
    };

    template<typename T>
    void SparseMatrix<T>::Builder::insert(index_type i, index_type j, const value_type &value) {
      assert_safe_numeric_limits();
      assert_index_is_valid(i, j);
      row_.push_back(i);
      col_.push_back(j);
      val_.push_back(value);
      is_sorted_ = false;
      is_merged_ = false;
    }

    template<typename T>
    void SparseMatrix<T>::Builder::sort() {
      if (is_sorted_) {
        return;
      }

      assert(row_.size() == col_.size());
      assert(row_.size() == val_.size());

      // We sort by row and column by finding the permutation and then applying
      // By doing both row and column in a single lambda we avoid doing two
      // separate stable_sort and apply_permutation giving a factor 2x speedup.
      std::vector<std::size_t> permutation(col_.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::sort(permutation.begin(), permutation.end(),
                [&](std::size_t i, std::size_t j) {
                    if (row_[i] < row_[j]) {
                      return true;
                    } else if (row_[i] == row_[j]) {
                      return col_[i] < col_[j];
                    }
                    return false;
                });

      row_ = apply_permutation(row_, permutation);
      col_ = apply_permutation(col_, permutation);
      val_ = apply_permutation(val_, permutation);

      assert(row_.size() == col_.size());
      assert(row_.size() == val_.size());

      is_sorted_ = true;
    }

    template<typename T>
    void SparseMatrix<T>::Builder::merge() {
      if (is_merged_) {
        return;
      }
      assert(row_.size() == col_.size());
      assert(row_.size() == val_.size());
      for (auto m = 1; m < row_.size(); ++m) {
        if (row_[m] == row_[m - 1] && col_[m] == col_[m - 1]) {
          val_[m - 1] += val_[m];
          val_.erase(val_.begin() + m);
          row_.erase(row_.begin() + m);
          col_.erase(col_.begin() + m);
        }
      }
      assert(row_.size() == col_.size());
      assert(row_.size() == val_.size());

      is_merged_ = true;
    }

    template<typename T>
    void SparseMatrix<T>::Builder::clear() {
      jams::util::force_deallocation(row_);
      jams::util::force_deallocation(col_);
      jams::util::force_deallocation(val_);
      is_sorted_ = false;
      is_merged_ = false;
    }

    template<typename T>
    SparseMatrix<T> SparseMatrix<T>::Builder::build() {
      switch (format_) {
        case SparseMatrixFormat::COO :
          return build_coo();
        case SparseMatrixFormat::CSR :
          return build_csr();
      };
      throw std::runtime_error("Unknown sparse matrix format for SparseMatrix<T>::Builder::build()");
    }

    template<typename T>
    SparseMatrix<T> SparseMatrix<T>::Builder::build_csr() {
      this->sort();
      this->merge();

      const auto nnz = val_.size();

      index_container csr_rows(num_rows_ + 1);

      csr_rows(0) = 0;
      index_type current_row = 0;
      index_type previous_row = 0;

      for (auto m = 1; m < nnz; ++m) {
        assert(m < row_.size());
        current_row = row_[m];
        if (current_row == previous_row) {
          continue;
        }

        assert(current_row + 1 < csr_rows.size());

        // fill in row array including any missing entries where there were no row,col values
        for (auto i = previous_row+1; i < current_row+1; ++i) {
          csr_rows(i) = m;
        }

        previous_row = current_row;
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
      this->merge();
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
      if (val_.size() >= std::numeric_limits<index_type>::max() - 1) {
        throw std::runtime_error("Number of non zero elements is too large for the sparse matrix index_type");
      }
    }

    template<typename T>
    void SparseMatrix<T>::Builder::assert_index_is_valid(SparseMatrix::index_type i, SparseMatrix::index_type j) const {
      if ((i >= num_rows_) || (i < 0) || (j >= num_cols_) || (j < 0)) {
        throw std::runtime_error("Invalid index for sparse matrix");
      }
    }

    template<typename T>
    void SparseMatrix<T>::Builder::output(std::ostream &os) {
      this->sort();
      this->merge();

      for (auto n = 0; n < row_.size(); ++n) {
        os << row_[n] << " " << col_[n] << " " << val_[n] << "\n";
      }
    }

    template<typename T>
    bool SparseMatrix<T>::Builder::is_structurally_symmetric() {
      this->sort();
      this->merge();

      for (auto n = 0; n < row_.size(); ++n) {
        auto i = row_[n];
        auto j = col_[n];

        auto low = std::lower_bound(row_.cbegin(), row_.cend(), j);

        // this j does not exist in row_ so matrix cannot be symmetric
        if ((*low) != j || low == row_.cend()) {
          // this col (j) does not exist in row_ so matrix cannot be structurally symmetric
          return false;
        }

        auto up = std::upper_bound(low, row_.cend(), j);

        auto col_begin = col_.cbegin() + (low - row_.cbegin());
        auto col_end = col_.cbegin() + (up - row_.cbegin());

        auto found = std::binary_search(col_begin, col_end, i);

        if (!found) {
          return false;
        }
      }
      return true;
    }

    template<typename T>
    bool SparseMatrix<T>::Builder::is_symmetric() {
      this->sort();
      this->merge();

      for (auto n = 0; n < row_.size(); ++n) {
        auto i = row_[n];
        auto j = col_[n];
        auto val = val_[n];

        auto low = std::lower_bound(row_.cbegin(), row_.cend(), j);

        if ((*low) != j || low == row_.cend()) {
          // this col (j) does not exist in row_ so matrix cannot be structurally symmetric
          return false;
        }

        auto up = std::upper_bound(low, row_.cend(), j);

        // all elements between 'low' and 'up' are data for row == j == col_[n]

        auto col_begin = col_.cbegin() + (low - row_.cbegin());
        auto col_end = col_.cbegin() + (up - row_.cbegin());

        auto col_loc = std::find(col_begin, col_end, i);

        if (col_loc == col_.cend()) {
          // this i does not exist in col_ so matrix cannot be structurally symmetric
          return false;
        }

        auto val_trans = val_.begin() + (col_loc - col_.cbegin());

        if (val != (*val_trans)) {
          // even though the matrix is structurally symmetric the values are not symmetric
          return false;
        }
      }
      return true;
    }
};

#endif // JAMS_CONTAINERS_SPARSE_MATRIX_BUILDER_H
