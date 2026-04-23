//
// Created by Joseph Barker on 2019-04-05.
//

#ifndef JAMS_MULTIARRAY_H
#define JAMS_MULTIARRAY_H

#include <jams/containers/synced_memory.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <tuple>
#include <type_traits>
#include <vector>

namespace jams {
    namespace detail {
        template<typename T, typename U, size_t i>
        constexpr auto array_cast(const std::array<U, i> &a) -> std::array<T, i> {
          std::array<T, i> result{};
          for (std::size_t n = 0; n < i; ++n) {
            result[n] = static_cast<T>(a[n]);
          }
          return result;
        }

        template<typename T, typename U>
        constexpr T checked_extent_cast(const U value) {
          static_assert(std::is_integral_v<T> && std::is_integral_v<U>,
                        "MultiArray extents must use integral types");

          if constexpr (std::is_signed_v<U>) {
            if (value < 0) {
              throw std::length_error("MultiArray extents must be non-negative");
            }
          }

          using unsigned_t = std::make_unsigned_t<T>;
          using unsigned_u = std::make_unsigned_t<U>;
          if (static_cast<unsigned_u>(value) > static_cast<unsigned_t>(std::numeric_limits<T>::max())) {
            throw std::length_error("MultiArray extents exceed size_type");
          }

          return static_cast<T>(value);
        }

        template<typename T, typename U, size_t i>
        constexpr auto checked_array_cast(const std::array<U, i> &a) -> std::array<T, i> {
          std::array<T, i> result{};
          for (std::size_t n = 0; n < i; ++n) {
            result[n] = checked_extent_cast<T>(a[n]);
          }
          return result;
        }

        template<typename T, std::size_t N>
        struct validated_shape {
          std::array<T, N> shape{};
          T elements{};
        };

        template<typename Value, typename Size, std::size_t N>
        struct validated_shape_with_value {
          validated_shape<Size, N> validated;
          Value value{};
        };

        template<typename Tuple, std::size_t... Is>
        constexpr bool tuple_elements_are_integral(std::index_sequence<Is...>) {
          return (std::is_integral_v<std::remove_reference_t<std::tuple_element_t<Is, Tuple>>> && ...);
        }

        template<std::size_t N, typename Value, typename Tuple, bool = (std::tuple_size_v<Tuple> == N + 1)>
        struct has_filled_shape_signature : std::false_type {};

        template<std::size_t N, typename Value, typename Tuple>
        struct has_filled_shape_signature<N, Value, Tuple, true>
            : std::bool_constant<tuple_elements_are_integral<Tuple>(std::make_index_sequence<N>{}) &&
                                 std::is_convertible_v<std::tuple_element_t<N, Tuple>, Value>> {};

        template<std::size_t N, typename Value, typename... Args>
        inline constexpr bool has_filled_shape_signature_v =
            has_filled_shape_signature<N, Value, std::tuple<Args...>>::value;

        template<typename T, std::size_t N>
        constexpr T element_count(const std::array<T, N> &shape) {
          static_assert(std::is_integral_v<T>, "MultiArray shape must use an integral size type");

          using unsigned_t = std::make_unsigned_t<T>;
          const auto max = static_cast<unsigned_t>(std::numeric_limits<T>::max());

          unsigned_t elements = 1;
          for (std::size_t n = 0; n < N; ++n) {
            const auto extent = static_cast<unsigned_t>(shape[n]);
            if (extent != 0 && elements > max / extent) {
              throw std::length_error("MultiArray element count exceeds size_type");
            }
            elements *= extent;
          }
          return static_cast<T>(elements);
        }

        template<typename T, typename... Args>
        constexpr validated_shape<T, sizeof...(Args)> make_validated_shape(const Args... args) {
          const auto shape = std::array<T, sizeof...(Args)>{{checked_extent_cast<T>(args)...}};
          return {shape, element_count(shape)};
        }

        template<typename T, typename U, std::size_t N>
        constexpr validated_shape<T, N> make_validated_shape(const std::array<U, N> &shape) {
          const auto validated = checked_array_cast<T>(shape);
          return {validated, element_count(validated)};
        }

        template<typename Value, typename Size, std::size_t N, typename Tuple, std::size_t... Is>
        constexpr validated_shape_with_value<Value, Size, N> make_validated_shape_with_value_from_tuple(
            Tuple&& args,
            std::index_sequence<Is...>) {
          return {make_validated_shape<Size>(std::get<Is>(args)...),
                  static_cast<Value>(std::get<N>(std::forward<Tuple>(args)))};
        }

        template<typename Value, typename Size, std::size_t N, typename... Args>
        constexpr validated_shape_with_value<Value, Size, N> make_validated_shape_with_value(const Args&... args) {
          static_assert(sizeof...(Args) == N + 1,
                        "number of MultiArray arguments does not match the filled-construction dimension");
          return make_validated_shape_with_value_from_tuple<Value, Size, N>(
              std::forward_as_tuple(args...),
              std::make_index_sequence<N>{});
        }

        template<typename T, std::size_t N>
        constexpr std::size_t
        row_major_index(const std::array<T, N> &dims, const std::array<T, N> &idx) {
          std::size_t offset = 0;
          for (std::size_t n = 0; n < N; ++n) {
            offset = offset * static_cast<std::size_t>(dims[n]) + static_cast<std::size_t>(idx[n]);
          }
          return offset;
        }

        template<typename T, std::size_t N, typename... Args>
        constexpr std::size_t
        row_major_index(const std::array<T, N> &dims, const Args &... args) {
          static_assert(sizeof...(args) == N,
                        "number of indices does not match the MultiArray dimension");
          return row_major_index(dims, std::array<T, N>{{static_cast<T>(args)...}});
        }

        template<typename T, std::size_t N>
        constexpr bool indices_in_bounds(const std::array<T, N> &dims,
                                         const std::array<T, N> &idx) {
          for (std::size_t n = 0; n < N; ++n) {
            if (!(idx[n] < dims[n])) {
              return false;
            }
          }
          return true;
        }

        template<std::size_t N, typename T, typename... Args>
        constexpr bool indices_in_bounds(const std::array<T, N> &dims, const Args &... args) {
          static_assert(sizeof...(args) == N,
                        "number of indices does not match the MultiArray dimension");
          return indices_in_bounds(dims, std::array<T, N>{{static_cast<T>(args)...}});
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
        template<class FTp_, std::size_t FDim_, class FIdx_>
        friend void zero(MultiArray<FTp_, FDim_, FIdx_>& x);

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

    private:
        using validated_shape_type = detail::validated_shape<size_type, Dim_>;
        using filled_shape_type = detail::validated_shape_with_value<value_type, size_type, Dim_>;

        struct validated_shape_tag {};
        struct filled_shape_tag {};
        struct range_construction_tag {};

        struct range_construction {
          validated_shape_type validated;
          SyncedMemory<Tp_> data;
        };

    public:
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
        template<typename... Args,
                 std::enable_if_t<(sizeof...(Args) == Dim_) && std::conjunction_v<std::is_integral<Args>...>, int> = 0>
        inline explicit MultiArray(const Args... args)
            : MultiArray(validated_shape_tag{}, detail::make_validated_shape<size_type>(args...)) {}

        // construct using dimensions followed by fill value
        template<typename... Args,
                 std::enable_if_t<detail::has_filled_shape_signature_v<Dim_, value_type, Args...>, int> = 0>
        inline explicit MultiArray(const Args&... args)
            : MultiArray(filled_shape_tag{}, detail::make_validated_shape_with_value<value_type, size_type, Dim_>(args...)) {}

        // construct using dimensions in array
        inline explicit MultiArray(const size_container_type& v)
            : MultiArray(validated_shape_tag{}, detail::make_validated_shape<size_type>(v)) {}

        inline explicit MultiArray(const size_container_type& v, const value_type& x)
            : MultiArray(filled_shape_tag{}, filled_shape_type{detail::make_validated_shape<size_type>(v), x}) {}

        template<typename Integral_,
                 std::enable_if_t<!std::is_same_v<std::remove_cv_t<Integral_>, size_type>, int> = 0>
        inline explicit MultiArray(const std::array<Integral_, Dim_> &v)
            : MultiArray(validated_shape_tag{}, detail::make_validated_shape<size_type>(v)) {}

        template<typename Integral_,
                 std::enable_if_t<!std::is_same_v<std::remove_cv_t<Integral_>, size_type>, int> = 0>
        inline explicit MultiArray(const std::array<Integral_, Dim_> &v, const value_type& x)
            : MultiArray(filled_shape_tag{}, filled_shape_type{detail::make_validated_shape<size_type>(v), x}) {}

        template<class InputIt,
                 std::enable_if_t<(Dim_ == 1) && !std::is_integral_v<InputIt>, int> = 0>
        inline MultiArray(InputIt first, InputIt last)
            : MultiArray(range_construction_tag{}, make_range_construction(first, last)) {}

        // capacity
        [[nodiscard]] inline constexpr bool empty() const noexcept {
          return data_.size() == 0;
        }

        template<std::size_t D = Dim_, std::enable_if_t<D == 1, int> = 0>
        [[nodiscard]] inline constexpr size_type size() const noexcept {
          return size_[0];
        }

        [[nodiscard]] inline constexpr size_type size(const size_type n) const noexcept {
          assert(n < Dim_);
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
        template<typename... Args,
                 std::enable_if_t<(sizeof...(Args) == Dim_) && std::conjunction_v<std::is_integral<Args>...>, int> = 0>
        inline reference operator()(const Args &... args) {
          assert(!empty());
          assert(detail::indices_in_bounds(size_, static_cast<size_type>(args)...));
          return data_.mutable_host_data()[detail::row_major_index(size_, static_cast<size_type>(args)...)];
        }

        template<typename... Args,
                 std::enable_if_t<(sizeof...(Args) == Dim_) && std::conjunction_v<std::is_integral<Args>...>, int> = 0>
        inline const_reference operator()(const Args &... args) const {
          assert(!empty());
          assert(detail::indices_in_bounds(size_, static_cast<size_type>(args)...));
          return data_.const_host_data()[detail::row_major_index(size_, static_cast<size_type>(args)...)];
        }

        inline reference operator()(const std::array<size_type, Dim_> &v) {
          assert(!empty());
          assert(detail::indices_in_bounds(size_, v));
          return data_.mutable_host_data()[detail::row_major_index(size_, v)];
        }

        inline const_reference operator()(const std::array<size_type, Dim_> &v) const {
          assert(!empty());
          assert(detail::indices_in_bounds(size_, v));
          return data_.const_host_data()[detail::row_major_index(size_, v)];
        }

        inline pointer data() {
          return data_.mutable_host_data();
        }

        inline const_pointer data() const {
          return data_.const_host_data();
        }

        inline const_pointer read_only_data() const {
          return data_.const_host_data();
        }

        inline pointer device_data() {
          return data_.mutable_device_data();
        }

        inline const_pointer device_data() const {
          return data_.const_device_data();
        }

        inline const_pointer read_only_device_data() const {
          return data_.const_device_data();
        }

        // iterators
        inline iterator begin() {
          return data_.mutable_host_data();
        }

        inline const_iterator begin() const {
          return data_.const_host_data();
        }

        inline const_iterator cbegin() const {
          return data_.const_host_data();
        }

        inline const_iterator read_only_begin() const {
          return data_.const_host_data();
        }

        inline iterator end() {
          pointer p = data_.mutable_host_data();
          return p ? p + data_.size() : p;
        }

        inline const_iterator end() const {
          const_pointer p = data_.const_host_data();
          return p ? p + data_.size() : p;
        }

        inline const_iterator cend() const {
          const_pointer p = data_.const_host_data();
          return p ? p + data_.size() : p;
        }

        inline const_iterator read_only_end() const {
          const_pointer p = data_.const_host_data();
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

        inline void fill(const value_type &value) {
          if (data_.size() == 0) {
            return;
          }
          pointer p = data_.mutable_host_data();
          std::fill(p, p + data_.size(), value);
        }

        template<typename... Args,
                 std::enable_if_t<(sizeof...(Args) == Dim_) && std::conjunction_v<std::is_integral<Args>...>, int> = 0>
        inline MultiArray& resize(const Args... args) {
          return resize_validated(detail::make_validated_shape<size_type>(args...));
        }

        inline MultiArray& resize(const size_container_type& v) {
          return resize_validated(detail::make_validated_shape<size_type>(v));
        }

        template<typename Integral_,
                 std::enable_if_t<!std::is_same_v<std::remove_cv_t<Integral_>, size_type>, int> = 0>
        inline MultiArray& resize(const std::array<Integral_, Dim_> &v) {
          return resize_validated(detail::make_validated_shape<size_type>(v));
        }

    private:
        inline explicit MultiArray(validated_shape_tag, const validated_shape_type& validated)
            : size_(validated.shape),
              data_(validated.elements) {}

        inline explicit MultiArray(filled_shape_tag, const filled_shape_type& filled)
            : size_(filled.validated.shape),
              data_(filled.validated.elements, filled.value) {}

        inline explicit MultiArray(range_construction_tag, range_construction range)
            : size_(range.validated.shape),
              data_(std::move(range.data)) {}

        inline MultiArray& resize_validated(const validated_shape_type& validated) {
          size_ = validated.shape;
          data_.resize(validated.elements);
          return *this;
        }

        template<class InputIt>
        static range_construction make_range_construction(InputIt first, InputIt last) {
          using iterator_category = typename std::iterator_traits<InputIt>::iterator_category;
          return make_range_construction(first, last, iterator_category{});
        }

        template<class InputIt>
        static range_construction make_range_construction(InputIt first,
                                                          InputIt last,
                                                          std::input_iterator_tag) {
          const std::vector<value_type> values(first, last);
          return {detail::make_validated_shape<size_type>(values.size()),
                  SyncedMemory<Tp_>(values.begin(), values.end())};
        }

        template<class ForwardIt>
        static range_construction make_range_construction(ForwardIt first,
                                                          ForwardIt last,
                                                          std::forward_iterator_tag) {
          const auto validated =
              detail::make_validated_shape<size_type>(std::distance(first, last));
          return {validated, SyncedMemory<Tp_>(first, last)};
        }

        size_container_type size_ = {{0}};
        mutable SyncedMemory<Tp_> data_;
    };

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

} // namespace jams

#include <jams/containers/multiarray_numeric.h>

#endif //JAMS_MULTIARRAY_H
