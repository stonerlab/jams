//
// Created by Joseph Barker on 2019-04-05.
//

#ifndef JAMS_MULTIARRAY_H
#define JAMS_MULTIARRAY_H

#include <jams/containers/synced_memory.h>

#include <array>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <type_traits>

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
            static constexpr std::size_t
            row_major(const std::array<std::size_t, N> &dims, const std::size_t &v) {
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
        constexpr std::size_t
        row_major_index(const std::array<std::size_t, N> &dims, const Args &... args) {
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

        static_assert(std::is_trivially_copyable<Tp_>::value,
              "MultiArray<T> requires trivially copyable T for device use");

        MultiArray() noexcept = default;
        ~MultiArray() = default;

        MultiArray(const MultiArray& rhs)
        : size_(rhs.size_), data_(rhs.data_) {}

        MultiArray(MultiArray&& rhs) noexcept
        : size_(std::move(rhs.size_)),
          data_(std::move(rhs.data_)) {}

        MultiArray& operator=(const MultiArray& rhs) & {
          size_ = rhs.size_;
          data_ = rhs.data_;
          return *this;
        }

        MultiArray& operator=(MultiArray&& rhs) & noexcept {
          size_ = std::move(rhs.size_);
          data_ = std::move(rhs.data_);
          return *this;
        }

        // construct using dimensions as arguments
        template<typename... Args, typename = std::enable_if_t<(std::conjunction_v<std::is_integral<Args>...> && (sizeof...(Args) == Dim_))>>
        inline explicit MultiArray(const Args... args) :
            size_({static_cast<size_type>(args)...}),
            data_(detail::product(static_cast<size_type>(args)...)) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in constructor does not match the MultiArray dimension");
        }

        template<typename... Args, typename = std::enable_if_t<(std::conjunction_v<std::is_integral<Args>...> && (sizeof...(Args) == Dim_))>>
        inline explicit MultiArray(const value_type& x, const Args... args):
            size_({static_cast<size_type>(args)...}),
            data_(detail::product(static_cast<size_type>(args)...), x) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in constructor does not match the MultiArray dimension");
        }

        // construct using dimensions in array
        template<typename Integral_>
        inline explicit MultiArray(const std::array<Integral_, Dim_> &v) :
            size_(detail::array_cast<size_type>(v)),
            data_(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(detail::array_cast<size_type>(v))) {}

      template<typename Integral_>
      inline explicit MultiArray(const value_type& x, const std::array<Integral_, Dim_> v) :
            size_(detail::array_cast<size_type>(v)),
            data_(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(detail::array_cast<size_type>(v)), x) {}

        // capacity
        [[nodiscard]] inline constexpr bool empty() const noexcept {
          return data_.size() == 0;
        }

        [[nodiscard]] inline constexpr size_type size(const size_type n) const noexcept {
          return size_[n];
        }

        [[nodiscard]] inline const size_container_type& shape() const noexcept {
          return size_;
        }

        [[nodiscard]] inline constexpr std::size_t bytes() const noexcept {
          return data_.bytes();
        }

        [[nodiscard]] inline constexpr size_type elements() const noexcept {
          return data_.size();
        }

        [[nodiscard]] inline constexpr dim_type dimension() const noexcept {
          return Dim_;
        }

        [[nodiscard]] inline constexpr size_type max_size() const noexcept {
          return data_.max_size();
        }

        // operations

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

        inline void swap(MultiArray& other) noexcept {
          using std::swap;
          swap(this->size_, other.size_);
          swap(this->data_, other.data_);
        }

        inline void zero() noexcept {
          data_.zero();
        }

        inline void fill(const value_type &value) {
          if (value == Tp_{0}) {
            zero();
            return;
          }
          pointer p = data_.mutable_host_data();
          std::fill(p, p + data_.size(), value);
        }

        template<typename... Args>
        inline MultiArray& resize(const Args &... args) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in resize does not match the MultiArray dimension");
          size_ = {static_cast<size_type>(args)...};
          data_.resize(detail::product(static_cast<size_type>(args)...));
          return *this;
        }

        inline MultiArray& resize(const std::array<size_type, Dim_> &v) {
          size_ = v;
          data_.resize(detail::vec<std::size_t, Dim_, Dim_>::last_n_product(v));
          return *this;
        }

    private:
        size_container_type size_ = { {0} };
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

        MultiArray() noexcept = default;
        ~MultiArray() = default;

        MultiArray(const MultiArray& rhs)
        : size_(rhs.size_), data_(rhs.data_){}

        MultiArray(MultiArray&& rhs) noexcept
        : size_(std::move(rhs.size_)), data_(std::move(rhs.data_)) {}

        MultiArray& operator=(const MultiArray& rhs) & {
          size_ = rhs.size_;
          data_ = rhs.data_;
          return *this;
        }

        MultiArray& operator=(MultiArray&& rhs) & noexcept {
          size_ = std::move(rhs.size_);
          data_ = std::move(rhs.data_);
          return *this;
        }

        inline explicit MultiArray(size_type size):
            size_({size}),
            data_(size) {}

        inline MultiArray(const Tp_& x, size_type size):
                size_({size}),
                data_(size, x) {}

        template <typename U>
        inline explicit MultiArray(const std::array<U, 1> &v) :
            size_(detail::array_cast<size_type>(v)),
            data_(std::get<0>(v)) {}

        template<class InputIt>
        inline MultiArray(InputIt first, InputIt last)
            : size_(detail::array_cast<size_type>(
                std::array<typename std::iterator_traits<InputIt>::difference_type,1>({std::distance(first, last)}))),
              data_(first, last) {}

        template <typename U>
        inline explicit MultiArray(const Tp_& x, const std::array<U, 1> &v) :
            size_(detail::array_cast<size_type>(v)),
            data_(std::get<0>(v), x) {}

        // capacity
        [[nodiscard]] inline constexpr bool empty() const noexcept {
          return data_.size() == 0;
        }

        [[nodiscard]] inline constexpr size_type size() const noexcept {
          return std::get<0>(size_);
        }

        [[nodiscard]] inline constexpr size_type size(const size_type n) const noexcept {
          static_assert(n == 0, "MultiArray.size(n) is greater than the dimension");
          return std::get<0>(size_);
        }

        [[nodiscard]] inline const size_container_type& shape() const noexcept {
          return size_;
        }

        [[nodiscard]] inline constexpr std::size_t bytes() const noexcept {
          return data_.bytes();
        }

        [[nodiscard]] inline constexpr size_type elements() const noexcept {
          return data_.size();
        }

        [[nodiscard]] inline constexpr dim_type dimension() const noexcept {
          return 1;
        }

        [[nodiscard]] inline constexpr size_type max_size() const noexcept {
          return data_.max_size();
        }

        // operations

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

        inline void swap(MultiArray& other) noexcept {
          using std::swap;
          swap(this->size_, other.size_);
          swap(this->data_, other.data_);
        }

        inline void zero() noexcept {
          data_.zero();
        }

        inline void fill(const value_type &value) {
          if (value == Tp_{0}) {
            zero();
            return;
          }
          pointer p = data_.mutable_host_data();
          std::fill(p, p + data_.size(), value);
        }

        inline MultiArray& resize( size_type count ) {
          std::get<0>(size_) = count;
          data_.resize(count);
          return *this;
        }

        inline MultiArray& resize(const std::array<size_type, 1> &v) {
          size_ = v;
          data_.resize(std::get<0>(v));
          return *this;
        }

    private:
        size_container_type size_ = { {0} };
        mutable SyncedMemory<Tp_> data_;
    };

    /**
    * Force a MultiArray to synchronise CPU and GPU data
    */
    template <typename T, std::size_t N>
    inline void force_multiarray_sync(const MultiArray<T,N> & x) {
      volatile const auto sync_data = x.data();
    }

    // allows simple zero(x.resize(a,b,c))
    template<class FTp_, std::size_t FDim_, class FIdx_>
    void zero(MultiArray<FTp_, FDim_, FIdx_>& x) {
      x.zero();
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    void fill(MultiArray<FTp_, FDim_, FIdx_>& x, const FTp_& y) {
      x.fill(y);
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

    template<class FTp_, std::size_t FDim_, class FIdx_, class Tp2_>
    inline void element_scale(MultiArray<FTp_, FDim_, FIdx_>& x, const Tp2_& y) {
      std::transform(x.begin(), x.end(), x.begin(), [y](const FTp_ &a) { return a * y; });
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    inline void element_sum(MultiArray<FTp_, FDim_, FIdx_>& x, const MultiArray<FTp_, FDim_, FIdx_>& y) {
      assert(x.elements() == y.elements());
      std::transform(y.begin(), y.end(), x.begin(), x.begin(),
                     [](const FTp_&x, const FTp_ &y) -> FTp_ { return x + y; });
    }


} // namespace jams

#endif //JAMS_MULTIARRAY_H
