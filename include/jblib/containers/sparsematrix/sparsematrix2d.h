#ifndef JBLIB_CONTAINERS_SPARSEMATRIX2D_H
#define JBLIB_CONTAINERS_SPARSEMATRIX2D_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <stdexcept>

#include "jblib/containers/sparsematrix/sparsematrix_template.h"
#include "jblib/containers/vec.h"

// A hash table would be good, but for simple hashes (like row-major) it is
// hard to have an integer type which will be large enough for to hash for
// really big sparse arrays. The problem is further compounded in larger dimensions
// like a 4D sparse matrix.

namespace jblib {

template <typename Tp_, int Component_>
  inline bool
  coordinate_compare(const std::pair<Vec2<int>, Tp_> &a, const std::pair<Vec2<int>, Tp_> &b) {
    return ( a.first[Component_] < b.first[Component_] );
  }

template <typename Tp_>
  class Sparsematrix<Tp_, 2> : public SparsematrixBase {
  private:
    std::vector<int> row_;
    std::vector<int> col_;
    std::vector<int> offsets_;
    std::vector<Tp_> values_;

    std::vector< std::pair<Vec2<int>, Tp_> > insertion_map_;

    bool map_;
    bool coo_;
    bool csr_;
    bool dia_;

  public:
    Sparsematrix(const int num_rows, const int num_cols) :
    SparsematrixBase(2),
    row_(),
    col_(),
    offsets_(),
    values_(),
    insertion_map_(),
    map_(true),
    coo_(false),
    csr_(false),
    dia_(false)
    { size(0) = num_rows; size(1) = num_cols; }

    // only ever allow const access to these
    const int* row() const { return &(row_[0]); }
    const int* col() const { return &(col_[0]); }
    const int* offsets() const {return &(offsets_[0]); }
    const Tp_* val() const { return &(values_[0]); }

    int num_rows() const { return size(0); }
    int num_cols() const { return size(1); }

    bool is_map_formatted() { return map_; }
    bool is_coo_formatted() { return coo_; }
    bool is_csr_formatted() { return csr_; }
    bool is_dia_formatted() { return dia_; }

    void set_map_formatted() { map_ = true; coo_ = csr_ = dia_ = false; }
    void set_coo_formatted() { coo_ = true; map_ = csr_ = dia_ = false; }
    void set_csr_formatted() { csr_ = true; map_ = coo_ = dia_ = false; }
    void set_dia_formatted() { dia_ = true; map_ = coo_ = csr_ = false; }

    void insert_value(const int i, const int j, const Tp_ value) {
      assert(is_map_formatted());
      // check indecies are inside matrix bounds
      if ((i >= num_rows()) || (i < 0)
       || (j >= num_cols()) || (j < 0)) {
        throw std::out_of_range("Attempted to insert a value out side of matrix size bounds in Sparsematrix");
    }

    if (is_new()) {
      set_open();
    }

    if (is_open()) {
      if (is_upper_storage() && i > j) {
        throw std::out_of_range("Attempted to inset into lower triangle of an upper Sparsematrix");
      }
      if (is_lower_storage() && i < j) {
        throw std::out_of_range("Attempted to inset into upper triangle of an lower Sparsematrix");
      }

      insertion_map_.push_back(std::pair<Vec2<int>, Tp_>(Vec2<int>(i,j), value));
      num_nonzeros()++;
    }
  }

  void insert_value_list(const int nz, const int *ivals, const int *jvals, const Tp_* values) {
    for (int i = 0; i < nz; ++i) {
      insert_value(ivals[i], jvals[i], values[i]);
    }
  }

  void insert_row(const int row, const int nz, const int *jvals, const Tp_* values) {
    for (int i = 0; i < nz; ++i) {
      insert_value(row, jvals[i], values[i]);
    }
  }

  void insert_col(const int col, const int nz, const int *ivals, const Tp_* values) {
    for (int i = 0; i < nz; ++i) {
      insert_value(ivals[i], col, values[i]);
    }
  }

  bool end_construction() {
      // sort the insertion map by row major (C) order
    std::stable_sort(insertion_map_.begin(), insertion_map_.end(), coordinate_compare<Tp_, 1>);
    std::stable_sort(insertion_map_.begin(), insertion_map_.end(), coordinate_compare<Tp_, 0>);
    return SparsematrixBase::end_construction();
  }

  void convert_map_to_coo() {
    if (!is_map_formatted()) {
      throw std::range_error("Attempted to convert Sparsematrix class format to coo from a non map format");
    }
    if (!is_valid()) {
      end_construction();
    }

    typename std::vector< std::pair<Vec2<int>, Tp_> >::const_iterator elem;

    int merged_value_counter = 0;

    // insert first element so that we can do the back checking below
    elem = insertion_map_.begin();
    row_.push_back(elem->first[0]);
    col_.push_back(elem->first[1]);
    values_.push_back(elem->second);
    merged_value_counter++;
    ++elem;

    for (; elem != insertion_map_.end(); ++elem) {
      if (elem->first[0] == row_.back() && elem->first[1] == col_.back()) {
        // merge (duplicate i,j values are summed)
        values_.back() += elem->second;
      } else {
        row_.push_back(elem->first[0]);
        col_.push_back(elem->first[1]);
        values_.push_back(elem->second);
        merged_value_counter++;
      }
    }

    // update num_nonzeros after the possible merging of duplicate i,j values
    num_nonzeros() = merged_value_counter;

    // clear insertion map memory
    (std::vector< std::pair<Vec2<int>, Tp_> >(0)).swap(insertion_map_);

    set_coo_formatted();
  }

  void convert_coo_to_dia() {
    // NOTE: internally, we guarantee coo to be sorted and merged (unique i,j values)
    if (!is_coo_formatted()) {
      throw std::range_error("Attempted to convert Sparsematrix class format to dia from a non coo format");
    }

    if (num_nonzeros() > 0) {
      num_diagonals() = 0;
      std::vector<bool> diagonal_has_values(num_rows() + num_cols(), false);
      std::vector<int>  diagonal_map(num_rows() + num_cols(), 0);

      // calculate which diagonals have values
      for (int i = 0; i < num_nonzeros(); ++i) {
        int map_index = (num_rows() - row_[i]) + col_[i];
        if (diagonal_has_values[map_index] == false) {
          diagonal_has_values[map_index] = true;
          num_diagonals()++;
        }
      }

      offsets_.resize(num_diagonals());
      std::vector<Tp_> dia_vals((num_rows()*num_diagonals()), 0.0);

      // calculate offsets from the leading diagonal
      for (int i = 0, diag = 0; i < (num_rows() + num_cols()); ++i) {
        if (diagonal_has_values[i]) {
          diagonal_map[i] = diag;
          offsets_[diag] = i - num_rows();
          diag++;
        }
      }

      // populate the values array
      for (int i = 0; i < num_nonzeros(); ++i) {
        int map_index = (num_rows() - row_[i]) + col_[i];
        int diag = diagonal_map[map_index];
        dia_vals[num_rows() * diag + row_[i]] = values_[i];
      }
      values_.swap(dia_vals);
    }
  }

  void convert_map_to_dia() {
    convert_map_to_coo();
    convert_coo_to_dia();
  }
};

}  // namespace jblib
#endif  // JBLIB_CONTAINERS_SPARSEMATRIX2D_H
