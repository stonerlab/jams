//
// Created by Joseph Barker on 2019-10-18.
//

#ifndef JAMS_CONTAINERS_INTERACTION_MATRIX_H
#define JAMS_CONTAINERS_INTERACTION_MATRIX_H

#include "jams/containers/multiarray.h"
#include "jams/containers/interaction_list.h"

namespace jams {
    template <typename T, typename BaseType>
    class InteractionMatrix {
    public:
        using value_type            = T;
        using value_reference       = value_type &;
        using const_value_reference = const value_type &;
        using value_pointer         = value_type *;
        using const_value_pointer   = const value_type *;

        using base_type             = BaseType;
        using base_reference        = base_type &;
        using const_base_reference  = const base_type &;
        using base_pointer          = base_type *;
        using const_base_pointer    = const base_type *;

        using size_type             = int;
        using size_reference        = size_type &;
        using const_size_reference  = const size_type &;
        using size_pointer          = size_type *;
        using const_size_pointer    = const size_type *;

        using index_container = MultiArray<size_type, 1>;
        using value_container = MultiArray<base_type, 1>;

        InteractionMatrix() = default;

        template <int N>
        InteractionMatrix(const InteractionList<value_type, N>& list, const int num_rows)
        : num_rows_(num_rows),
        num_vals_(list.value_table_.size()),
        num_interactions_(list.size()),
        index_size_(N),
        val_size_(sizeof(value_type) / sizeof(base_type)),
        row_(num_rows_ + 1),
        indices_(num_interactions_ * index_size_),
        val_(num_vals_ * val_size_) {
          // double check value_type is an integer multiple of base_type
          assert (sizeof(value_type) % sizeof(base_type) == 0);
          assert(index_size_ >= 1);
          assert(val_size_ >= 1);

          // populate row_ (CSR style)
          {
            row_(0) = 0;
            size_type r = 0;
            int i = 0;
            while (r < num_rows) {
              while (i < num_interactions_) {
                const auto row_index = list.indicies_[i][0];
                if (row_index != r) {
                  break;
                }
                i++;
              }
              row_(r + 1) = i;
              r++;
            }
          }

          // populate indicies_ (data is j,k,l, ..., value_key)
          for (auto m = 0; m < num_interactions_; ++m) {
            // first index is used for row_
            for (auto n = 1; n < index_size_; ++n) {
              assert(list.indicies_[m][n] < num_rows_);
              assert(index_size_ * m + (n - 1) < indices_.size());
              indices_(index_size_ * m + (n - 1)) = list.indicies_[m][n];
            }
            // last index is lookup key for val_
            indices_(index_size_ * m + index_size_ - 1) = list.value_lookup_[m];
          }

          // populate vals_
          for (auto m = 0; m < num_vals_; ++m) {
            // first index is used for row_
            for (auto n = 0; n < val_size_; ++n) {
              val_(val_size_ * m + n) = list.value_table_[m][n];
            }
          }
        }

        inline std::size_t memory() const {
          return row_.bytes() + indices_.bytes() + val_.bytes();
        }

        inline constexpr int num_rows() const { return num_rows_; }
        inline constexpr int num_vals() const { return num_vals_; }
        inline constexpr int num_interactions() const { return num_interactions_; }
        inline constexpr int index_size() const { return index_size_; }
        inline constexpr int val_size() const { return val_size_; }

        inline const_size_pointer row_data()  const { return row_.data(); }
        inline const_size_pointer index_data()  const { return indices_.data(); }
        inline const_base_pointer val_data()  const { return val_.data(); }

        inline const_size_pointer row_device_data()  const { return row_.device_data(); }
        inline const_size_pointer index_device_data()  const { return indices_.device_data(); }
        inline const_base_pointer val_device_data()  const { return val_.device_data(); }

    private:
        int num_rows_ = 0;
        int num_vals_ = 0;
        int num_interactions_ = 0;
        int index_size_ = 1;
        int val_size_ = 1;
        index_container row_;
        index_container indices_;
        value_container val_;
    };
}

#endif //JAMS_CONTAINERS_INTERACTION_MATRIX_H
