//
// Created by Joseph Barker on 2019-04-05.
//
// MultiArray is a fixed-rank, row-major array owning SyncedMemory storage.
//
// The public API separates logical reads from writes:
//   - const accessors and host_view() synchronize data to the host but do not
//     mark host storage modified.
//   - mutable_host_data(), mutable_host_view(), non-const begin(), and
//     non-const operator() expose writable host storage and mark device storage
//     stale through SyncedMemory.
//   - device_data() returns a const device pointer; device writes must use
//     mutable_device_data().
//
// Prefer operator() for concise scalar access and host views for tight loops.
// A host view captures the synchronized host pointer, shape, and row-major
// strides once:
//
//   const auto s = spins.host_view();
//   for (std::size_t i = 0; i < s.extent(0); ++i) {
//     const auto* row = s.row_data(i);
//     // read row[0], row[1], ...
//   }
//
//   auto out = field.mutable_host_view();
//   out(i, j) = value;
//
// Construction and resize validate extents and total element count. Element
// access is assert-only for speed: out-of-bounds access is undefined in release
// builds.

#ifndef JAMS_MULTIARRAY_H
#define JAMS_MULTIARRAY_H

#include <jams/containers/synced_memory.h>

#include <array>
#include <algorithm>
#include <cassert>
#include <concepts>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace jams {
    namespace detail {
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
        constexpr std::array<T, i> checked_array_cast_helper(const std::array<U, i> &a, std::index_sequence<Is...>) {
          return {{checked_integral_cast<T>(std::get<Is>(a))...}};
        }

        template<typename T, typename U, size_t i>
        constexpr auto checked_array_cast(const std::array<U, i> &a) -> std::array<T, i> {
          return checked_array_cast_helper<T>(a, std::make_index_sequence<i>{});
        }

        template<typename T, std::size_t N>
        constexpr std::size_t checked_product(const std::array<T, N>& v) {
          const auto max_size = static_cast<std::size_t>(std::numeric_limits<T>::max());
          std::size_t result = 1;
          for (const auto value : v) {
            const auto extent = checked_integral_cast<std::size_t>(value);
            if (extent != 0 && result > std::numeric_limits<std::size_t>::max() / extent) {
              throw std::overflow_error("MultiArray shape product overflow");
            }
            result *= extent;
            if (result > max_size) {
              throw std::overflow_error("MultiArray shape product exceeds size_type");
            }
          }
          return result;
        }

        template<typename Size, std::size_t N, std::size_t... Is>
        constexpr std::size_t row_major_index_array_impl(const std::array<Size, N> &dims,
                                                         const std::array<Size, N> &idx,
                                                         std::index_sequence<Is...>) {
          std::size_t offset = 0;
          ((offset = offset * static_cast<std::size_t>(std::get<Is>(dims)) +
                     static_cast<std::size_t>(std::get<Is>(idx))), ...);
          return offset;
        }

        template<typename Size, std::size_t N>
        constexpr std::size_t row_major_index(const std::array<Size, N> &dims,
                                              const std::array<Size, N> &idx) {
          return row_major_index_array_impl(dims, idx, std::make_index_sequence<N>{});
        }

        template<typename T>
        concept multiarray_extent =
            std::integral<std::remove_cvref_t<T>> &&
            !std::same_as<std::remove_cvref_t<T>, bool>;

        template<std::size_t Dim, typename... Args>
        concept valid_extent_args =
            sizeof...(Args) == Dim && (multiarray_extent<Args> && ...);

        template<typename T, std::size_t Dim>
        struct is_multiarray_extent_array : std::false_type {};

        template<typename Integral, std::size_t ArrayDim, std::size_t Dim>
        struct is_multiarray_extent_array<std::array<Integral, ArrayDim>, Dim>
            : std::bool_constant<ArrayDim == Dim && multiarray_extent<Integral>> {};

        template<typename T, std::size_t Dim>
        inline constexpr bool is_multiarray_extent_array_v =
            is_multiarray_extent_array<std::remove_cvref_t<T>, Dim>::value;

        template<multiarray_extent Size>
        [[nodiscard]] constexpr bool index_in_bounds(const Size index, const Size extent) noexcept {
          return !std::cmp_less(index, 0) && std::cmp_less(index, extent);
        }

        template<typename Size, std::size_t Dim, typename... Args>
        requires valid_extent_args<Dim, Args...>
        constexpr std::array<Size, Dim> make_size_container(Args... args) {
          return {checked_integral_cast<Size>(args)...};
        }

        template<typename Size, typename Integral, std::size_t Dim>
        requires multiarray_extent<Integral>
        constexpr std::array<Size, Dim> make_size_container(const std::array<Integral, Dim>& values) {
          return checked_array_cast<Size>(values);
        }

        template<typename Size, std::size_t Dim>
        constexpr std::array<std::size_t, Dim> row_major_strides(const std::array<Size, Dim>& shape) noexcept {
          std::array<std::size_t, Dim> strides{};
          std::size_t stride = 1;
          for (std::size_t i = Dim; i > 0; --i) {
            strides[i - 1] = stride;
            stride *= static_cast<std::size_t>(shape[i - 1]);
          }
          return strides;
        }

        template<typename Size, std::size_t Dim, std::size_t... Is>
        constexpr std::size_t row_major_index_from_strides_impl(const std::array<std::size_t, Dim>& strides,
                                                                const std::array<Size, Dim>& idx,
                                                                std::index_sequence<Is...>) noexcept {
          std::size_t offset = 0;
          ((offset += strides[Is] * static_cast<std::size_t>(idx[Is])), ...);
          return offset;
        }

        template<typename Size, std::size_t Dim>
        constexpr std::size_t row_major_index_from_strides(const std::array<std::size_t, Dim>& strides,
                                                           const std::array<Size, Dim>& idx) noexcept {
          return row_major_index_from_strides_impl(strides, idx, std::make_index_sequence<Dim>{});
        }
    }

    /**
     * Lightweight row-major view of MultiArray host storage.
     *
     * MultiArrayHostView does not own memory. It stores a pointer, the shape,
     * row-major strides, and total element count captured when it is created by
     * MultiArray::host_view() or MultiArray::mutable_host_view().
     *
     * Use host_view() for read-only host access. It is available on both const
     * and non-const MultiArray objects and does not mark host storage modified.
     * Use mutable_host_view() when writing on the host; it marks device storage
     * stale once when the view is created instead of once per scalar access.
     *
     * Views are invalidated by operations that can replace or resize the owning
     * MultiArray storage, including resize(), clear(), move assignment, and
     * destruction of the owner. They are also only coherent with the memory side
     * they were created for; do not keep a view across later host/device
     * mutations of the owning MultiArray.
     *
     * Bounds checking is assert-only. In release builds operator() and
     * row_data() assume valid indices.
     */
    template<class Tp_, std::size_t Dim_, class Idx_ = std::size_t>
    class MultiArrayHostView {
    public:
        /// Element type without cv-qualification.
        using value_type = std::remove_const_t<Tp_>;
        /// Integral type used for extents and indices.
        using size_type = Idx_;
        /// Type used for dimension numbers.
        using dim_type = std::size_t;
        /// Fixed-size shape container, one extent per dimension.
        using size_container_type = std::array<size_type, Dim_>;
        /// Fixed-size row-major stride container, one stride per dimension.
        using stride_container_type = std::array<std::size_t, Dim_>;
        /// Pointer to host data. Const views expose const pointers.
        using pointer = Tp_*;
        /// Const pointer to host data.
        using const_pointer = const value_type*;
        /// Reference to a host element. Const views expose const references.
        using reference = Tp_&;
        /// Iterator over host data. Const views expose const iterators.
        using iterator = pointer;
        /// Read-only iterator over host data.
        using const_iterator = const_pointer;
        /// Span over host data. Const views expose read-only spans.
        using span_type = std::span<Tp_>;

        static_assert(Dim_ > 0, "MultiArrayHostView dimension must be greater than zero");
        static_assert(std::is_integral_v<Idx_>, "MultiArrayHostView index type must be integral");

        constexpr MultiArrayHostView() noexcept = default;

        constexpr MultiArrayHostView(pointer data, size_container_type shape, size_type elements) noexcept
            : data_(data),
              shape_(shape),
              strides_(detail::row_major_strides(shape)),
              elements_(elements) {}

        /// Return the captured host pointer. May be nullptr for an empty view.
        [[nodiscard]] constexpr pointer data() const noexcept {
          return data_;
        }

        /// Return the total number of elements.
        [[nodiscard]] constexpr size_type size() const noexcept {
          return elements_;
        }

        /// Return the total number of elements.
        [[nodiscard]] constexpr size_type elements() const noexcept {
          return elements_;
        }

        /// Return true when the view contains no elements.
        [[nodiscard]] constexpr bool empty() const noexcept {
          return elements_ == 0;
        }

        /// Return the extent of dimension n.
        [[nodiscard]] constexpr size_type extent(const dim_type n) const noexcept {
          assert(n < Dim_);
          return shape_[n];
        }

        /// Return all extents.
        [[nodiscard]] constexpr const size_container_type& shape() const noexcept {
          return shape_;
        }

        /// Return a flat span over the captured host storage.
        [[nodiscard]] constexpr span_type flat_span() const noexcept {
          return span_type(data_, static_cast<std::size_t>(elements_));
        }

        /// Return an iterator to the first captured host element.
        [[nodiscard]] constexpr iterator begin() const noexcept {
          return data_;
        }

        /// Return an iterator one past the last captured host element.
        [[nodiscard]] constexpr iterator end() const noexcept {
          return data_ ? data_ + static_cast<std::size_t>(elements_) : data_;
        }

        /// Return a read-only iterator to the first captured host element.
        [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
          return data_;
        }

        /// Return a read-only iterator one past the last captured host element.
        [[nodiscard]] constexpr const_iterator cend() const noexcept {
          return data_ ? data_ + static_cast<std::size_t>(elements_) : data_;
        }

        /// Return the row-major stride of dimension n, in elements.
        [[nodiscard]] constexpr std::size_t stride(const dim_type n) const noexcept {
          assert(n < Dim_);
          return strides_[n];
        }

        /// Return a reference to an element using Dim_ indices.
        template<typename... Args>
        [[nodiscard]] constexpr reference operator()(const Args&... args) const noexcept {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArrayHostView indicies does not match the dimension");
          const size_container_type indices{static_cast<size_type>(args)...};
          assert(indices_in_bounds(indices));
          return data_[detail::row_major_index_from_strides(strides_, indices)];
        }

        /// Return a reference to an element using an index array.
        [[nodiscard]] constexpr reference operator()(const size_container_type& indices) const noexcept {
          assert(indices_in_bounds(indices));
          return data_[detail::row_major_index_from_strides(strides_, indices)];
        }

        /**
         * Return a pointer to the first element at index n in dimension 0.
         *
         * For a 2D row-major array this is the start of row n. For higher-rank
         * arrays it is the start of the contiguous block with first index n.
         */
        [[nodiscard]] constexpr pointer row_data(const size_type n) const noexcept {
          static_assert(Dim_ >= 1, "row_data requires at least one dimension");
          assert(detail::index_in_bounds(n, shape_[0]));
          return data_ ? data_ + static_cast<std::size_t>(n) * strides_[0] : data_;
        }

        /**
         * Return a span over the contiguous block at index n in dimension 0.
         *
         * For a 2D row-major array this is row n. For higher-rank arrays it is
         * the contiguous block with first index n.
         */
        [[nodiscard]] constexpr span_type row_span(const size_type n) const noexcept {
          return span_type(row_data(n), strides_[0]);
        }

    private:
        [[nodiscard]] constexpr bool indices_in_bounds(const size_container_type& indices) const noexcept {
          for (dim_type dim = 0; dim < Dim_; ++dim) {
            if (!detail::index_in_bounds(indices[dim], shape_[dim])) {
              return false;
            }
          }
          return true;
        }

        pointer data_ = nullptr;
        size_container_type shape_ = { {0} };
        stride_container_type strides_ = { {0} };
        size_type elements_ = 0;
    };

    /**
     * Fixed-rank row-major array with synchronized host/device storage.
     *
     * MultiArray owns a contiguous SyncedMemory<T> buffer and a fixed-size shape
     * array. The rank is part of the type, while extents are runtime values.
     * Element type T must be trivially copyable because storage may be copied to
     * and from CUDA device memory.
     *
     * Indexing order is row-major:
     *
     *   offset = (((i0 * extent(1) + i1) * extent(2) + i2) ...)
     *
     * Common usage:
     *
     *   jams::MultiArray<double, 2> a(num_spins, 3);
     *   a(i, 0) = sx;
     *
     * For repeated host access, prefer views:
     *
     *   auto a_host = a.mutable_host_view();
     *   auto* row = a_host.row_data(i);
     *   row[0] = sx;
     *
     * The scalar operator() path is intentionally lightweight. It performs
     * assert-only bounds checks and does not throw on invalid indices in release
     * builds. Construction and resize do validate extents and total size.
     *
     * Synchronization model:
     *   - host_data(), data(), begin() const, and host_view() read host storage.
     *   - host_data() const and host_view() may copy from device to host.
     *   - mutable_host_data(), mutable_host_view(), non-const begin(), and
     *     non-const operator() mark host storage modified.
     *   - device_data() reads device storage and returns a const pointer.
     *   - mutable_device_data() marks device storage modified.
     */
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
        using span_type = std::span<value_type>;
        using const_span_type = std::span<const value_type>;
        using host_view_type = MultiArrayHostView<value_type, Dim_, size_type>;
        using const_host_view_type = MultiArrayHostView<const value_type, Dim_, size_type>;

        static_assert(std::is_trivially_copyable_v<Tp_>,
              "MultiArray<T> requires trivially copyable T for device use");
        static_assert(Dim_ > 0,
              "MultiArray dimension must be greater than zero");
        static_assert(std::is_integral_v<Idx_>,
              "MultiArray index type must be integral");
        static_assert(!std::is_same_v<std::remove_cv_t<Idx_>, bool>,
              "MultiArray index type must not be bool");

        MultiArray() noexcept = default;
        ~MultiArray() = default;

        MultiArray(const MultiArray& rhs)
        : size_(rhs.size_), data_(rhs.data_) {}

        MultiArray(MultiArray&& rhs) noexcept
        : size_(std::exchange(rhs.size_, {})),
          data_(std::move(rhs.data_)) {}

        MultiArray& operator=(const MultiArray& rhs) & {
          data_ = rhs.data_;
          size_ = rhs.size_;
          return *this;
        }

        MultiArray& operator=(MultiArray&& rhs) & noexcept {
          if (this != &rhs) {
            size_ = std::exchange(rhs.size_, {});
            data_ = std::move(rhs.data_);
          }
          return *this;
        }

        // Construct using dimensions as arguments.
        template<typename... Args>
        requires detail::valid_extent_args<Dim_, Args...>
        inline explicit MultiArray(const Args... args)
            : MultiArray(detail::make_size_container<size_type, Dim_>(args...)) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in constructor does not match the MultiArray dimension");
        }

        template<typename... Args>
        requires detail::valid_extent_args<Dim_, Args...>
        inline explicit MultiArray(const value_type& x, const Args... args)
            : MultiArray(x, detail::make_size_container<size_type, Dim_>(args...)) {
          static_assert(sizeof...(args) == Dim_,
                        "number of MultiArray indicies in constructor does not match the MultiArray dimension");
        }

        // Construct using dimensions in an array.
        template<typename Integral_>
        requires detail::multiarray_extent<Integral_>
        inline explicit MultiArray(const std::array<Integral_, Dim_> &v) :
            size_(detail::make_size_container<size_type>(v)),
            data_(detail::checked_product(size_)) {}

      template<typename Integral_>
      requires detail::multiarray_extent<Integral_>
      inline explicit MultiArray(const value_type& x, const std::array<Integral_, Dim_> v) :
            size_(detail::make_size_container<size_type>(v)),
            data_(detail::checked_product(size_), x) {}

        template<std::input_iterator InputIt>
        requires (Dim_ == 1)
        inline MultiArray(InputIt first, InputIt last)
            : data_(first, last) {
          size_ = {detail::checked_integral_cast<size_type>(data_.size())};
        }

        template<std::ranges::input_range Range>
        requires (Dim_ == 1 &&
                  std::convertible_to<std::ranges::range_reference_t<Range>, value_type> &&
                  !detail::is_multiarray_extent_array_v<Range, Dim_>)
        inline explicit MultiArray(Range&& values)
            : data_(std::forward<Range>(values)) {
          size_ = {detail::checked_integral_cast<size_type>(data_.size())};
        }

        // Capacity and shape.
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
          const auto max_count = std::min<std::size_t>(
              data_.max_size(),
              static_cast<std::size_t>(std::numeric_limits<size_type>::max()));
          return static_cast<size_type>(max_count);
        }

        // Element access.
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

        /**
         * Return a read-only span over host storage.
         *
         * This may synchronize device data to host memory. It does not mark host
         * storage modified, even when called on a non-const MultiArray.
         */
        [[nodiscard]] inline const_span_type host_span() {
          return data_.host_span();
        }

        /**
         * Return a read-only span over host storage.
         *
         * This may synchronize device data to host memory. It does not mark host
         * storage modified.
         */
        [[nodiscard]] inline const_span_type host_span() const {
          return data_.host_span();
        }

        /**
         * Return a writable span over host storage.
         *
         * This marks host storage modified and device storage stale once when
         * the span is created.
         */
        [[nodiscard]] inline span_type mutable_host_span() {
          return data_.mutable_host_span();
        }

        /**
         * Return a read-only host view.
         *
         * This may synchronize device data to host memory. It does not mark host
         * storage modified, even when called on a non-const MultiArray.
         */
        inline const_host_view_type host_view() {
          return const_host_view_type(data_.host_data(), size_, data_.size());
        }

        /**
         * Return a read-only host view.
         *
         * This may synchronize device data to host memory. It does not mark host
         * storage modified.
         */
        inline const_host_view_type host_view() const {
          return const_host_view_type(data_.host_data(), size_, data_.size());
        }

        /**
         * Return a writable host view.
         *
         * This marks host storage modified and device storage stale once when
         * the view is created.
         */
        inline host_view_type mutable_host_view() {
          return host_view_type(data_.mutable_host_data(), size_, data_.size());
        }

        inline const_pointer device_data() {
          return data_.device_data();
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
          std::ranges::fill(mutable_host_span(), value);
        }

        template<typename... Args>
        requires detail::valid_extent_args<Dim_, Args...>
        inline MultiArray& resize(const Args &... args) {
          const auto new_size = detail::make_size_container<size_type, Dim_>(args...);
          data_.resize(detail::checked_product(new_size));
          size_ = new_size;
          return *this;
        }

        template<typename Integral_>
        requires detail::multiarray_extent<Integral_>
        inline MultiArray& resize(const std::array<Integral_, Dim_> &v) {
          const auto new_size = detail::make_size_container<size_type>(v);
          data_.resize(detail::checked_product(new_size));
          size_ = new_size;
          return *this;
        }

    private:
        [[nodiscard]] bool indices_in_bounds(const size_container_type& indices) const noexcept {
          for (dim_type dim = 0; dim < Dim_; ++dim) {
            if (!detail::index_in_bounds(indices[dim], size_[dim])) {
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
      auto values = x.mutable_host_span();
      std::ranges::transform(values, values.begin(), [y](const FTp_ &a) { return a * y; });
    }

    template<class FTp_, std::size_t FDim_, class FIdx_>
    inline void element_sum(MultiArray<FTp_, FDim_, FIdx_>& x, const MultiArray<FTp_, FDim_, FIdx_>& y) {
      assert(x.elements() == y.elements());
      auto x_values = x.mutable_host_span();
      const auto y_values = y.host_span();
      std::ranges::transform(y_values, x_values, x_values.begin(),
                             [](const FTp_& y_value, const FTp_& x_value) -> FTp_ {
                               return x_value + y_value;
                             });
    }


} // namespace jams

#endif //JAMS_MULTIARRAY_H
