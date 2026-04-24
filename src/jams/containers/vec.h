//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_VEC_H
#define JAMS_VEC_H

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "jams/helpers/mixed_precision.h"

namespace jams {

template <typename T, std::size_t N>
struct Vec {
  static_assert(N > 0, "Vec requires at least one component");

  using value_type = T;
  using size_type = std::size_t;
  using storage_type = std::array<T, N>;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;

  storage_type values{};

  constexpr Vec() = default;
  constexpr Vec(const Vec&) = default;
  constexpr Vec(Vec&&) = default;
  constexpr Vec& operator=(const Vec&) = default;
  constexpr Vec& operator=(Vec&&) = default;
  ~Vec() = default;

  constexpr Vec(const storage_type& storage) : values(storage) {}
  constexpr Vec(storage_type&& storage) : values(std::move(storage)) {}

  template <typename... Args,
            typename = std::enable_if_t<(sizeof...(Args) > 0 && sizeof...(Args) <= N) &&
                                        (std::is_convertible_v<Args, T> && ...)>>
  constexpr Vec(Args&&... args) : values{static_cast<T>(std::forward<Args>(args))...} {}

  constexpr reference operator[](size_type i) noexcept { return values[i]; }
  constexpr const_reference operator[](size_type i) const noexcept { return values[i]; }

  constexpr pointer data() noexcept { return values.data(); }
  constexpr const_pointer data() const noexcept { return values.data(); }

  constexpr iterator begin() noexcept { return values.begin(); }
  constexpr const_iterator begin() const noexcept { return values.begin(); }
  constexpr const_iterator cbegin() const noexcept { return values.cbegin(); }

  constexpr iterator end() noexcept { return values.end(); }
  constexpr const_iterator end() const noexcept { return values.end(); }
  constexpr const_iterator cend() const noexcept { return values.cend(); }

  static constexpr size_type size() noexcept { return N; }
  static constexpr bool empty() noexcept { return false; }

  constexpr operator storage_type&() noexcept { return values; }
  constexpr operator const storage_type&() const noexcept { return values; }
};

template <typename T, std::size_t N>
inline constexpr bool operator<(const Vec<T, N>& lhs, const Vec<T, N>& rhs) {
  return lhs.values < rhs.values;
}

template <typename T, std::size_t N>
inline constexpr bool operator>(const Vec<T, N>& lhs, const Vec<T, N>& rhs) {
  return rhs < lhs;
}

template <typename T, std::size_t N>
inline constexpr bool operator<=(const Vec<T, N>& lhs, const Vec<T, N>& rhs) {
  return !(rhs < lhs);
}

template <typename T, std::size_t N>
inline constexpr bool operator>=(const Vec<T, N>& lhs, const Vec<T, N>& rhs) {
  return !(lhs < rhs);
}

static_assert(sizeof(Vec<double, 3>) == sizeof(double) * 3,
              "Vec must not add storage padding");
static_assert(std::is_trivially_copyable_v<Vec<double, 3>>,
              "Vec must be trivially copyable");
static_assert(std::is_standard_layout_v<Vec<double, 3>>,
              "Vec must use standard layout storage");

using Vec2 = Vec<double, 2>;

using Vec3 = Vec<double, 3>;
using Vec3f = Vec<float, 3>;
using Vec3R = Vec<Real, 3>;
using Vec3b = Vec<bool, 3>;
using Vec3i = Vec<int, 3>;
using Vec3cx = Vec<std::complex<double>, 3>;

using Vec4 = Vec<double, 4>;
using Vec4i = Vec<int, 4>;

} // namespace jams

template <typename T, std::size_t N>
using Vec = jams::Vec<T, N>;

using Vec2 = jams::Vec2;

using Vec3 = jams::Vec3;
using Vec3f = jams::Vec3f;
using Vec3R = jams::Vec3R;
using Vec3b = jams::Vec3b;
using Vec3i = jams::Vec3i;
using Vec3cx = jams::Vec3cx;

using Vec4 = jams::Vec4;
using Vec4i = jams::Vec4i;

#endif // JAMS_VEC_H
