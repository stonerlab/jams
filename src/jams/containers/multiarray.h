//
// Created by Joseph Barker on 2019-04-05.
//

#ifndef JAMS_MULTIARRAY_H
#define JAMS_MULTIARRAY_H

#include <jams/containers/synced_memory.h>

#include <array>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

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

        template<typename To, typename From>
        constexpr To checked_integral_cast(From value) {
          static_assert(std::is_integral_v<To>, "target type must be integral");
          static_assert(std::is_integral_v<From>, "source type must be integral");
          static_assert(!std::is_same_v<std::remove_cv_t<To>, bool>, "bool extents are not supported");
          static_assert(!std::is_same_v<std::remove_cv_t<From>, bool>, "bool extents are not supported");

          if constexpr (std::is_signed_v<From>) {
            if (value < 0) {
              throw std::length_error("MultiArray extent must be non-negative");
            }
          }

          using common_type = std::common_type_t<std::make_unsigned_t<From>, std::make_unsigned_t<To>, std::size_t>;
          const auto unsigned_value = static_cast<common_type>(value);
          const auto max_value = static_cast<common_type>(std::numeric_limits<To>::max());
          if (unsigned_value > max_value) {
            throw std::overflow_error("MultiArray extent conversion overflow");
          }
          return static_cast<To>(value);
        }

        template<typename T, typename U, size_t i, size_t... Is>
        constexpr std::array<T, i> checked_array_cast_helper(const std::array<U, i> &a, indices<Is...>) {
          return {{checked_integral_cast<T>(std::get<Is>(a))...}};
        }

        template<typename T, typename U, size_t i>
        constexpr auto checked_array_cast(const std::array<U, i> &a) -> std::array<T, i> {
          return checked_array_cast_helper<T>(a, build_indices<i>());
        }

        template<typename T, std::size_t N>
        constexpr std::size_t checked_product(const std::array<T, N>& v) {
          std::size_t result = 1;
          for (const auto value : v) {
            const auto extent = checked_integral_cast<std::size_t>(value);
            if (extent != 0 && result > std::numeric_limits<std::size_t>::max() / extent) {
              throw std::overflow_error("MultiArray shape product overflow");
            }
            result *= extent;
          }
          return result;
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

        template<typename Size, std::size_t N, std::size_t... Is>
        constexpr std::size_t row_major_index_array_impl(const std::array<Size, N> &dims,
                                                         const std::array<Size, N> &idx,
                                                         indices<Is...>) {
          std::size_t offset = 0;
          ((offset = offset * static_cast<std::size_t>(std::get<Is>(dims)) +
                     static_cast<std::size_t>(std::get<Is>(idx))), ...);
          return offset;
        }

        template<typename Size, std::size_t N>
        constexpr std::size_t row_major_index(const std::array<Size, N> &dims,
                                              const std::array<Size, N> &idx) {
          return row_major_index_array_impl(dims, idx, build_indices<N>());
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
        friend void swap(MultiArray<FTp_, FDim_, FIdx_>& lhs, MultiArray<FTp_, FDim_, FIdx_>& rhs) noexcept;

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
        static_assert(std::is_integral<Idx_>::value,
              "MultiArray index type must be integral");
        static_assert(!std::is_same_v<std::remove_cv_t<Idx_>, bool>,
              "MultiArray index type must not be bool");

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
        inline explicit MultiArray(const Args... args)
            : MultiArray(size_container_type{detail::checked_integral_cast<size_type>(args)...}) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in constructor does not match the MultiArray dimension");
        }

        template<typename... Args, typename = std::enable_if_t<(std::conjunction_v<std::is_integral<Args>...> && (sizeof...(Args) == Dim_))>>
        inline explicit MultiArray(const value_type& x, const Args... args)
            : MultiArray(x, size_container_type{detail::checked_integral_cast<size_type>(args)...}) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in constructor does not match the MultiArray dimension");
        }

        // construct using dimensions in array
        template<typename Integral_>
        inline explicit MultiArray(const std::array<Integral_, Dim_> &v) :
            size_(detail::checked_array_cast<size_type>(v)),
            data_(detail::checked_product(size_)) {}

      template<typename Integral_>
      inline explicit MultiArray(const value_type& x, const std::array<Integral_, Dim_> v) :
            size_(detail::checked_array_cast<size_type>(v)),
            data_(detail::checked_product(size_), x) {}

        template<class InputIt, std::enable_if_t<(Dim_ == 1 && detail::is_iterator<InputIt>::value), bool> = true>
        inline MultiArray(InputIt first, InputIt last)
            : data_(first, last) {
          size_ = {static_cast<size_type>(data_.size())};
        }

        // capacity
        [[nodiscard]] inline constexpr bool empty() const noexcept {
          return data_.size() == 0;
        }

        [[nodiscard]] inline constexpr size_type size() const noexcept {
          return data_.size();
        }

        [[nodiscard]] inline constexpr size_type extent(const size_type n) const noexcept {
          assert(n < Dim_);
          return size_[n];
        }

        [[nodiscard]] inline const size_container_type& shape() const noexcept {
          return size_;
        }

        [[nodiscard]] inline constexpr std::size_t bytes() const {
          return data_.bytes();
        }

        [[nodiscard]] inline constexpr size_type elements() const noexcept {
          return data_.size();
        }

        [[nodiscard]] inline constexpr dim_type rank() const noexcept {
          return Dim_;
        }

        [[nodiscard]] inline size_type max_size() const {
          return data_.max_size();
        }

        // operations

        // element access
        template<typename... Args>
        inline reference operator()(const Args &... args) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies does not match the MultiArray dimension");
          const size_container_type indices{static_cast<size_type>(args)...};
          assert(indices_in_bounds(indices));
          return data_.mutable_host_data()[detail::row_major_index(size_, indices)];
        }

        template<typename... Args>
        inline const_reference operator()(const Args &... args) const {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies does not match the MultiArray dimension");
          const size_container_type indices{static_cast<size_type>(args)...};
          assert(indices_in_bounds(indices));
          return data_.host_data()[detail::row_major_index(size_, indices)];
        }

        inline reference operator()(const std::array<size_type, Dim_> &v) {
          assert(indices_in_bounds(v));
          return data_.mutable_host_data()[detail::row_major_index(size_, v)];
        }

        inline const_reference operator()(const std::array<size_type, Dim_> &v) const {
          assert(indices_in_bounds(v));
          return data_.host_data()[detail::row_major_index(size_, v)];
        }

        inline pointer data() {
          return host_data();
        }

        inline const_pointer data() const {
          return host_data();
        }

        inline pointer host_data() {
          return data_.mutable_host_data();
        }

        inline const_pointer host_data() const {
          return data_.host_data();
        }

        inline pointer device_data() {
          return mutable_device_data();
        }

        inline const_pointer device_data() const {
          return data_.device_data();
        }

        inline pointer mutable_device_data() {
          return data_.mutable_device_data();
        }

        // iterators
        inline iterator begin() {
          return data_.mutable_host_data();
        }

        inline const_iterator begin() const {
          return data_.host_data();
        }

        inline const_iterator cbegin() const {
          return data_.host_data();
        }

        inline iterator end() {
          pointer p = begin();
          return p ? p + data_.size() : p;
        }

        inline const_iterator end() const {
          const_pointer p = begin();
          return p ? p + data_.size() : p;
        }

        inline const_iterator cend() const {
          const_pointer p = cbegin();
          return p ? p + data_.size() : p;
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

        inline void zero() {
          data_.zero();
        }

        inline void release_stale_host() noexcept {
          data_.release_stale_host();
        }

        inline void release_stale_device() noexcept {
          data_.release_stale_device();
        }

        inline void fill(const value_type &value) {
          if (empty()) {
            return;
          }
          if constexpr (detail::synced_memory_byte_zeroable_v<value_type>) {
            if (value == value_type{}) {
              zero();
              return;
            }
          }
          std::fill_n(data_.mutable_host_data(), data_.size(), value);
        }

        template<typename... Args>
        inline MultiArray& resize(const Args &... args) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in resize does not match the MultiArray dimension");
          const size_container_type new_size{detail::checked_integral_cast<size_type>(args)...};
          data_.resize(detail::checked_product(new_size));
          size_ = new_size;
          return *this;
        }

        inline MultiArray& resize(const std::array<size_type, Dim_> &v) {
          data_.resize(detail::checked_product(v));
          size_ = v;
          return *this;
        }

    private:
        [[nodiscard]] bool indices_in_bounds(const size_container_type& indices) const noexcept {
          using unsigned_size_type = std::make_unsigned_t<size_type>;
          for (dim_type dim = 0; dim < Dim_; ++dim) {
            if constexpr (std::is_signed_v<size_type>) {
              if (indices[dim] < 0) {
                return false;
              }
            }
            if (static_cast<unsigned_size_type>(indices[dim]) >= static_cast<unsigned_size_type>(size_[dim])) {
              return false;
            }
          }
          return true;
        }

        size_container_type size_ = { {0} };
        SyncedMemory<Tp_> data_;
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
    void swap(MultiArray<FTp_, FDim_, FIdx_>& lhs, MultiArray<FTp_, FDim_, FIdx_>& rhs) noexcept {
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
