//
// Created by Joseph Barker on 2019-10-16.
//

#ifndef JAMS_INTERACTION_LIST_H
#define JAMS_INTERACTION_LIST_H

#include "jams/containers/vector_set.h"
#include "jams/containers/unordered_vector_set.h"

namespace jams {
    // This is not neccesarily the most efficient method (especially for a  2D interaction list) but is quite general
    // to allow higher order interactions (e.g. 3 or 4 indicies). Memory use is minimized by only storing unique
    // values for the interactions.
    template<class T, int N>
    class InteractionList {
        template<typename U, typename BaseType>
        friend class InteractionMatrix;
    public:
        using value_type = T;
        using size_type = int;
        using index_type = int;
        using index_array_type = std::array<int, N>;
        using pair_type = std::pair<const index_array_type&, const value_type&>;

        InteractionList()  = default;

        void insert(const index_array_type& index, const T &value) {
          auto insert_pos = indicies_.insert_and_get_position(index);
          assert(insert_pos >= 0);
          assert(insert_pos < indicies_.size() + 1);

          auto value_pos = value_table_.insert_and_get_position(value);
          assert(value_pos >= 0);
          assert(value_pos < value_table_.size() + 1);

          // now put the lookup index into the same position as indicies_
          value_lookup_.insert(value_lookup_.begin() + insert_pos, value_pos);
        }

        inline bool contains(const index_array_type& index) const {
          return indicies_.find(index) != indicies_.end();
        }

        pair_type operator[] (const size_type i) const {
          return {indicies_[i], value_table_[value_lookup_[i]]};
        }

        inline size_type size() const {
          return indicies_.size();
        }

        inline size_type memory() const {
          // we use size() rather than capacity() because the system can always reduce the capacity to free some memory
          return indicies_.size() * sizeof(index_array_type) + value_lookup_.size() * sizeof(size_type)
                 + value_table_.size() * sizeof(T);
        }

        inline size_type num_interactions(const index_type i) const {
          auto range = first_index_position_range(i);
          return range.second - range.first;
        }

        std::vector<pair_type> interactions_of(const index_type i) const {
          auto range = first_index_position_range(i);
          std::vector<pair_type> result;
          result.reserve(range.second - range.first);

          for (auto n = range.first; n < range.second; ++n) {
            result.push_back((*this)[n]);
          }
          return result;
        }


    private:

        std::pair<size_type, size_type> first_index_position_range(const index_type i) const {
          struct Comp {
              bool operator() ( const index_array_type &x, index_type i ) const { return std::get<0>(x) < i; }
              bool operator() ( index_type i, const index_array_type& x ) const { return i < std::get<0>(x); }
          };

          auto range = std::equal_range(std::begin(indicies_), std::end(indicies_), i, Comp{});
          return std::make_pair(
              std::distance(begin(indicies_), range.first), std::distance(begin(indicies_), range.second));
        }

        jams::VectorSet<index_array_type> indicies_;
        std::vector<size_type> value_lookup_;
        jams::UnorderedVectorSet<T> value_table_;
    };
}

#endif //JAMS_INTERACTION_LIST_H
