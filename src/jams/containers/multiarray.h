//
// Created by Joseph Barker on 2019-04-05.
//

#ifndef JAMS_MULTIARRAY_H
#define JAMS_MULTIARRAY_H

#include <array>
#include <cassert>
#include <cstring>

#include "jams/containers/synced_memory.h"

namespace jams {
    namespace detail {
        template <std::size_t... Is>
        struct indices {};
        template <std::size_t N, std::size_t... Is>
        struct build_indices: build_indices<N-1, N-1, Is...> {};
        template <std::size_t... Is>
        struct build_indices<0, Is...>: indices<Is...> {};

        template<typename T, typename U, size_t i, size_t... Is>
        constexpr std::array<T, i> array_cast_helper(const std::array<U, i> &a, indices<Is...>) {
          return {{static_cast<T>(std::get<Is>(a))...}};
        }

        template<typename T, typename U, size_t i>
        constexpr auto array_cast(const std::array<U, i> &a) -> std::array<T, i> {
          // tag dispatch to helper with array indices
          return array_cast_helper<T>(a, build_indices<i>());
        }

        // partial specialization of templates is not possible, so we use structs

        // recursive method to multiply the last N elements of array
        template<typename T, std::size_t N, std::size_t I>
        struct vec {
            static constexpr T last_n_product(const std::array<T, N> &v) {
              return std::get<N - I>(v) * vec<T, N, I - 1>::last_n_product(v);
            }
        };

        template<typename T, std::size_t N>
        struct vec<T, N, 0> {
            static constexpr T last_n_product(const std::array<T, N> &v) {
              return 1;
            }
        };

        // recursive methods to generate row major arg_indices at compile time
        template<std::size_t N, std::size_t I>
        struct arg_indices {
            template<typename... Args>
            static constexpr std::size_t
            row_major(const std::array<std::size_t, N> &dims, const std::size_t &first, const Args &... args) {
              return first * vec<std::size_t, N, I - 1>::last_n_product(dims) +
                     arg_indices<N, I - 1>::row_major(dims, args...);
            }
        };

        template<std::size_t N>
        struct arg_indices<N, 1> {
            static constexpr std::size_t row_major(const std::array<std::size_t, N> &dims, const std::size_t &v) {
              return v;
            }
        };

        template<std::size_t N, std::size_t I>
        struct arr_indices {
            static constexpr std::size_t
            row_major(const std::array<std::size_t, N> &dims, const std::array<std::size_t, N> &idx) {
              return std::get<I - 1>(idx) + std::get<I - 1>(dims) * arr_indices<N, I - 1>::row_major(dims, idx);
            }
        };

        template<std::size_t N>
        struct arr_indices<N, 1> {
            static constexpr std::size_t
            row_major(const std::array<std::size_t, N> &dims, const std::array<std::size_t, N> &idx) {
              return std::get<0>(idx);
            }
        };

        template<std::size_t N, typename... Args>
        constexpr std::size_t row_major_index(const std::array<std::size_t, N> &dims, const Args &... args) {
          return arg_indices<N, N>::row_major(dims, args...);
        }

        template<std::size_t N>
        constexpr std::size_t
        row_major_index(const std::array<std::size_t, N> &dims, const std::array<std::size_t, N> &idx) {
          return arr_indices<N, N>::row_major(dims, idx);
        }

        template<typename T, typename... Args>
        constexpr T product(T v) {
          return v;
        }

        template<typename T, typename... Args>
        constexpr T product(T first, Args... args) {
          return first * product(args...);
        }
    }

    template<class Tp_, std::size_t Dim_, class Idx_ = std::size_t>
    class MultiArray {
    public:
        template<class FTp_, std::size_t FDim_, class FIdx_>
        friend void swap(MultiArray<FTp_, FDim_, FIdx_>& lhs, MultiArray<FTp_, FDim_, FIdx_>& rhs);

        using value_type = Tp_;
        using size_type  = Idx_;
        using dim_type   = std::size_t;
        using size_container_type = std::array<size_type, Dim_>;
        using difference_type = std::ptrdiff_t;
        using reference  = value_type &;
        using const_reference = const value_type &;
        using pointer = value_type *;
        using const_pointer = const value_type *;
        using iterator = pointer;
        using const_iterator = const_pointer;

        MultiArray() = default;
        ~MultiArray() = default;

        MultiArray(const MultiArray& other):
          size_(other.size_), data_(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(other.size_)){
          std::copy(other.begin(), other.end(), this->begin());
        }

        // construct using dimensions as arguments
        template<typename... Args>
        inline explicit MultiArray(const Args... args):
            size_({static_cast<size_type>(args)...}),
            data_(detail::product(static_cast<size_type>(args)...)) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in constructor does not match the MultiArray dimension");
        }

        // construct using dimensions in array
        template <typename U>
        inline explicit MultiArray(const std::array<U, Dim_> &v) :
            size_(detail::array_cast<size_type>(v)),
            data_(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(detail::array_cast<size_type>(v))) {}

        // construct using dimensions in array and initial value
        template <typename U>
        inline explicit MultiArray(const std::array<U, Dim_> &v, const Tp_ &x) :
            size_(detail::array_cast<size_type>(v)),
            data_(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(detail::array_cast<size_type>(v)), x) {}

        // capacity
        inline constexpr bool empty() const noexcept {
          return data_.size() == 0;
        }

        inline constexpr size_type size(const size_type n) const noexcept {
          return size_[n];
        }

        inline const size_container_type& shape() const noexcept {
          return size_;
        }

        inline constexpr size_type memory() const noexcept {
          return data_.memory();
        }

        inline constexpr size_type elements() const noexcept {
          return data_.size();
        }

        inline constexpr dim_type dimension() const noexcept {
          return Dim_;
        }

        inline constexpr size_type max_size() const noexcept {
          return data_.max_size();
        }

        // operations
        inline void fill(const value_type &value) {
          std::fill(data_.mutable_host_data(), data_.mutable_host_data() + data_.size(), value);
        }

        // element access
        template<typename... Args>
        inline reference operator()(const Args &... args) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies does not match the MultiArray dimension");
          assert(!empty());
          return data_.mutable_host_data()[detail::row_major_index(size_, static_cast<size_type>(args)...)];
        }

        template<typename... Args>
        inline const_reference operator()(const Args &... args) const {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies does not match the MultiArray dimension");
          assert(!empty());
          return data_.const_host_data()[detail::row_major_index(size_, static_cast<size_type>(args)...)];
        }

        inline reference operator()(const std::array<size_type, Dim_> &v) {
          assert(!empty());
          return data_.mutable_host_data()[detail::row_major_index(size_, v)];
        }

        inline const_reference operator()(const std::array<size_type, Dim_> &v) const {
          assert(!empty());
          return data_.const_host_data()[detail::row_major_index(size_, v)];
        }

        inline MultiArray& operator=(const MultiArray& other) {
          if (this == &other) return *this;

          if (size_ != other.size_) {
            size_ = other.size_;
            data_.resize(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(other.size_));
          }

          std::copy(other.begin(), other.end(), this->begin());

          return *this;
        }

        inline pointer data() noexcept {
          return data_.mutable_host_data();
        }

        inline const_pointer data() const noexcept {
          return data_.const_host_data();
        }

        inline pointer device_data() noexcept {
          return data_.mutable_device_data();
        }

        inline const_pointer device_data() const noexcept {
          return data_.const_device_data();
        }

        // iterators
        inline iterator begin() noexcept {
          return data_.mutable_host_data();
        }

        inline const_iterator begin() const noexcept {
          return data_.const_host_data();
        }

        inline const_iterator cbegin() const noexcept {
          return data_.const_host_data();
        }

        inline iterator end() noexcept {
          return data_.mutable_host_data() + data_.size();
        }

        inline const_iterator end() const noexcept {
          return data_.const_host_data() + data_.size();
        }

        inline const_iterator cend() const noexcept {
          return data_.const_host_data() + data_.size();
        }

        // Modifiers
        inline void clear() noexcept {
          size_.fill(0);
          data_.clear();
        }

        inline void zero() noexcept {
          data_.zero();
        }

        template<typename... Args>
        inline void resize(const Args &... args) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in resize does not match the MultiArray dimension");
          size_ = {static_cast<size_type>(args)...},
          data_.resize(detail::product(static_cast<size_type>(args)...));
        }

        inline void resize(const std::array<size_type, Dim_> &v) {
          size_ = v;
          data_.resize(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(v));
        }

    private:
        size_container_type size_ = {0};
        mutable SyncedMemory<Tp_> data_;
    };

    // specialize for 1D
    template<class Tp_, class Idx_>
    class MultiArray<Tp_, 1, Idx_> {
    public:
        using value_type = Tp_;
        using size_type  = Idx_;
        using dim_type   = std::size_t;
        using size_container_type = std::array<size_type, 1>;
        using difference_type = std::ptrdiff_t;
        using reference  = value_type &;
        using const_reference = const value_type &;
        using pointer = value_type *;
        using const_pointer = const value_type *;
        using iterator = pointer;
        using const_iterator = const_pointer;

        template<class FTp_, std::size_t FDim_, class FIdx_>
        friend void swap(MultiArray<FTp_, FDim_, FIdx_>& lhs, MultiArray<FTp_, FDim_, FIdx_>& rhs);

        MultiArray() = default;
        ~MultiArray() = default;

        MultiArray(const MultiArray& other):
            size_(other.size_), data_(std::get<0>(other.size_)){
          std::copy(other.begin(), other.end(), this->begin());
        }

        inline explicit MultiArray(size_type size):
            size_({size}),
            data_(size) {}

        inline explicit MultiArray(size_type size, const Tp_& x):
            size_({size}),
            data_(size) { fill(x); }

        template <typename U>
        inline explicit MultiArray(const std::array<U, 1> &v) :
            size_(detail::array_cast<size_type>(v)),
            data_(std::get<0>(v)) {}

        template <typename U>
        inline explicit MultiArray(const std::array<U, 1> &v, const Tp_& x) :
            size_(detail::array_cast<size_type>(v)),
            data_(std::get<0>(v)) { fill(x); }

        // capacity
        inline constexpr bool empty() const noexcept {
          return data_.size() == 0;
        }

        inline constexpr size_type size() const noexcept {
          return std::get<0>(size_);
        }

        inline constexpr size_type size(const size_type n) const noexcept {
          return std::get<0>(size_);
        }

        inline const size_container_type& shape() const noexcept {
          return size_;
        }

        inline constexpr size_type memory() const noexcept {
          return data_.memory();
        }

        inline constexpr size_type elements() const noexcept {
          return data_.size();
        }

        inline constexpr dim_type dimension() const noexcept {
          return 1;
        }

        inline constexpr size_type max_size() const noexcept {
          return data_.max_size();
        }

        // operations
        inline void fill(const value_type &value) {
          if (value == value_type{0}) {
            data_.zero();
            return;
          }
          std::fill(data_.mutable_host_data(), data_.mutable_host_data() + data_.size(), value);
        }

        // element access
        inline reference operator()(const size_type & x) {
          assert(!empty() && x < data_.size());
          return data_.mutable_host_data()[x];
        }

        inline const_reference operator()(const size_type & x) const {
          assert(!empty() && x < data_.size());
          return data_.const_host_data()[x];
        }

        inline reference operator()(const std::array<size_type, 1> &v) {
          assert(!empty() && std::get<0>(v) < data_.size());
          return data_.mutable_host_data()[std::get<0>(v)];
        }

        inline const_reference operator()(const std::array<size_type, 1> &v) const {
          assert(!empty() && std::get<0>(v) < data_.size());
          return data_.const_host_data()[std::get<0>(v)];
        }

        inline MultiArray& operator=(const MultiArray& other) {
          if (this == &other) return *this;

          if (size_ != other.size_) {
            size_ = other.size_;
            data_.resize(detail::vec<std::size_t, 1, 1>::last_n_product(other.size_));
          }

          std::copy(other.begin(), other.end(), this->begin());

          return *this;
        }

        inline pointer data() noexcept {
          return data_.mutable_host_data();
        }

        inline const_pointer data() const noexcept {
          return data_.const_host_data();
        }

        inline pointer device_data() noexcept {
          return data_.mutable_device_data();
        }

        inline const_pointer device_data() const noexcept {
          return data_.const_device_data();
        }

        // iterators
        inline iterator begin() noexcept {
          return data_.mutable_host_data();
        }

        inline const_iterator begin() const noexcept {
          return data_.const_host_data();
        }

        inline const_iterator cbegin() const noexcept {
          return data_.const_host_data();
        }

        inline iterator end() noexcept {
          return data_.mutable_host_data() + data_.size();
        }

        inline const_iterator end() const noexcept {
          return data_.const_host_data() + data_.size();
        }

        inline const_iterator cend() const noexcept {
          return data_.const_host_data() + data_.size();
        }

        // Modifiers
        inline void clear() noexcept {
          size_.fill(0);
          data_.clear();
        }

        inline void zero() noexcept {
          data_.zero();
        }

        inline void resize( size_type count ) {
          std::get<0>(size_) = count;
          data_.resize(count);
        }

        inline void resize( size_type count, const value_type& value ) {
          std::get<0>(size_) = count;
          data_.resize(count);
          fill(value);
        }

        inline void resize(const std::array<size_type, 1> &v) {
          size_ = v;
          data_.resize(std::get<0>(v));
        }

    private:
        size_container_type size_ = {0};
        mutable SyncedMemory<Tp_> data_;
    };

    /**
    * Force a MultiArray to synchronise CPU and GPU data
    */
    template <typename T, std::size_t N>
    inline void force_multiarray_sync(const MultiArray<T,N> & x) {
      volatile const auto sync_data = x.data();
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    void swap(MultiArray<FTp_, FDim_, FIdx_>& lhs, MultiArray<FTp_, FDim_, FIdx_>& rhs) {
      using std::swap;
      swap(lhs.size_, rhs.size_);
      swap(lhs.data_, rhs.data_);
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    typename MultiArray<FTp_, FDim_, FIdx_>::iterator
    inline begin(MultiArray<FTp_, FDim_, FIdx_>& x) {
      return x.begin();
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    typename MultiArray<FTp_, FDim_, FIdx_>::const_iterator
    inline begin(const MultiArray<FTp_, FDim_, FIdx_>& x) {
      return x.begin();
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    typename MultiArray<FTp_, FDim_, FIdx_>::iterator
    inline end(MultiArray<FTp_, FDim_, FIdx_>& x) {
      return x.end();
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    typename MultiArray<FTp_, FDim_, FIdx_>::const_iterator
    inline end(const MultiArray<FTp_, FDim_, FIdx_>& x) {
      return x.end();
    }


} // namespace jams

#endif //JAMS_MULTIARRAY_H
