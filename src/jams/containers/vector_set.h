//
// Created by Joseph Barker on 2019-10-16.
//

#ifndef JAMS_VECTOR_SET_H
#define JAMS_VECTOR_SET_H

#include <functional>
#include <algorithm>
#include <vector>

// See http://lafstern.org/matt/col1.pdf

namespace jams {
    template <class T, class Compare = std::less<T>, class Allocator = std::allocator<T>>
    class VectorSet {
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

        explicit VectorSet(const Compare& c = Compare())
            : data_(), cmp_(c) {}

        template <class InputIterator>
        VectorSet(InputIterator first, InputIterator last, const Compare& c = Compare())
        : data_(first, last), cmp_(c) {
          std::sort(begin(), end(), cmp_);
        }

        std::pair<iterator,bool> insert(const T& t) {
          iterator it = lower_bound(begin(), end(), t, cmp_);
          if (it == end() || cmp_(t, *it)) {
            data_.insert(it, t);
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

        const_iterator find(const T& t) const {
          const_iterator it = lower_bound(begin(), end(), t, cmp_);
          return it == end() || cmp_(t, *it) ? end() : it;
        }

        inline reference       operator[]( size_type pos ) { return data_[pos]; }
        inline const_reference operator[]( size_type pos ) const { return data_[pos]; }

        inline pointer data() noexcept { return data_.data(); }
        inline const_pointer data() const noexcept { return data_.data(); }

        inline reference back() { return data_.back(); }
        inline const_reference back() const { return data_.back(); }

        inline reference front() { return data_.front(); }
        inline const_reference front() const { return data_.front(); }

        size_type size() const noexcept { return data_.size(); }
        size_type capacity() const noexcept { return data_.capacity(); }

        inline iterator begin() noexcept { return data_.begin(); }
        inline const_iterator begin() const noexcept { return data_.begin(); }
        inline const_iterator cbegin() const noexcept { return data_.cbegin(); }

        inline iterator end() noexcept { return data_.end(); }
        inline const_iterator end() const noexcept { return data_.end(); }
        inline const_iterator cend() const noexcept { return data_.cend(); }

    private:
        Compare cmp_;
        container_type data_;
    };
};

#endif //JAMS_VECTOR_SET_H
