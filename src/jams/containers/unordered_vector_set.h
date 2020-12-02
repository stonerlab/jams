//
// Created by Joseph Barker on 2019-10-16.
//

#ifndef JAMS_UNORDERED_VECTOR_SET_H
#define JAMS_UNORDERED_VECTOR_SET_H

#include <algorithm>
#include <vector>

namespace jams {
    template <class T, class Allocator = std::allocator<T>>
    class UnorderedVectorSet {
    public:
        using value_type = T;
        using container_type = std::vector<T, Allocator>;
        using allocator_type = Allocator;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;
        using pointer = typename container_type::pointer;
        using const_pointer = typename container_type::const_pointer;
        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;
        using reverse_iterator = typename container_type::reverse_iterator;
        using const_reverse_iterator = typename container_type::const_reverse_iterator;

        inline reference       operator[]( size_type pos ) { return data_[pos]; }
        inline const_reference operator[]( size_type pos ) const { return data_[pos]; }

        inline pointer data() noexcept { return data_.data(); }
        inline const_pointer data() const noexcept { return data_.data(); }

        size_type size() const noexcept { return data_.size(); }
        size_type capacity() const noexcept { return data_.capacity(); }

        std::pair<iterator,bool> insert(const T& x) {
          iterator it = std::find(begin(), end(), x);
          if (it == end()) {
            data_.push_back(x);
            return {it, true};
          }
          return {it, false};
        }

        // returns the position (index of vector) of the inserted object
        typename std::iterator_traits<iterator>::difference_type
        insert_and_get_position(const T& x) {
          // iterators are invalid after insert so we need to get begin() before,
          // the returned iterator is in terms of the original iterator range
          iterator it_begin = begin();
          iterator it = insert(x).first;
          return std::distance(it_begin, it);
        }

        inline iterator begin() noexcept { return data_.begin(); }
        inline const_iterator begin() const noexcept { return data_.begin(); }
        inline const_iterator cbegin() const noexcept { return data_.cbegin(); }

        inline iterator end() noexcept { return data_.end(); }
        inline const_iterator end() const noexcept { return data_.end(); }
        inline const_iterator cend() const noexcept { return data_.cend(); }

    private:
        container_type data_;
    };
};

#endif //JAMS_UNORDERED_VECTOR_SET_H
