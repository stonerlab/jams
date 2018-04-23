//
// Created by Joe Barker on 2018/04/20.
//

#ifndef JAMS_CSR_H
#define JAMS_CSR_H

#include <array>
#include <vector>
#include <map>

template <class Tp_>
class CSRMatrix {
public:
    using value_type = Tp_;
    using size_type  = std::size_t ;
    using size_container_type = std::array<size_type, 2>;
    using data_container_type = std::vector<std::map<size_type, value_type >>;
    using difference_type = std::ptrdiff_t;
    using reference  = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using iterator = typename data_container_type::iterator;
    using const_iterator = typename data_container_type::const_iterator;
    using reverse_iterator = typename data_container_type::reverse_iterator;
    using const_reverse_iterator = typename data_container_type::const_reverse_iterator;

    CSRMatrix() = default;
    ~CSRMatrix() = default;

    inline explicit CSRMatrix(const size_type i, const size_type j) :
            size_({i,j}),
            data_(i * j) {}

    // capacity
    inline constexpr bool empty() const noexcept {
      return data_.empty();
    }

    inline constexpr size_type size(const size_type n) const noexcept {
      //assert(n < Dim_);
      return size_[n];
    }

    inline constexpr size_type elements() const noexcept {
      return data_.size();
    }

    bool exists(const size_type i, const size_type j) {
      return data_[i].count(j);
    }

    const_reference operator()(const size_type i, const size_type j) const {
      if (exists(i, j)) {
        return data_[i][j];
      }
      return 0.0;
    }

    reference operator()(const size_type i, const size_type j) {
      return data_[i][j];
    }

private:
    size_container_type size_ = {0};
    data_container_type data_;
};

#endif //JAMS_CSR_H
