//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_VEC_H
#define JAMS_VEC_H

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace jams {

template <typename T, std::size_t N>
struct Vec;

template <typename T, std::size_t Rows, std::size_t Cols>
struct Mat;

template <typename>
struct is_mat : std::false_type {};

template <typename>
struct is_vec : std::false_type {};

template <typename T, std::size_t N>
struct is_vec<Vec<T, N>> : std::true_type {};

namespace detail {

template <typename To, typename From, typename = void>
struct is_static_castable : std::false_type {};

template <typename To, typename From>
struct is_static_castable<To, From,
                          std::void_t<decltype(static_cast<To>(std::declval<From>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_container_scalar_v =
    !is_vec<std::decay_t<T>>::value && !is_mat<std::decay_t<T>>::value;

template <typename To, typename From, std::size_t N, std::size_t... I>
constexpr Vec<To, N> array_cast_impl(const Vec<From, N>& in, std::index_sequence<I...>);

template <typename T, std::size_t N, std::size_t... I>
constexpr auto unary_minus_impl(const Vec<T, N>& rhs, std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto scalar_times_vec_impl(const T1& lhs, const Vec<T2, N>& rhs,
                                     std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_times_scalar_impl(const Vec<T1, N>& lhs, const T2& rhs,
                                     std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_div_scalar_impl(const Vec<T1, N>& lhs, const T2& rhs,
                                   std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto scalar_div_vec_impl(const T1& lhs, const Vec<T2, N>& rhs,
                                   std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_plus_vec_impl(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs,
                                 std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_minus_vec_impl(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs,
                                  std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_mod_vec_impl(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs,
                                std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto dot_impl(const Vec<T1, N>& a, const Vec<T2, N>& b,
                        std::index_sequence<I...>);

template <typename T, std::size_t N, std::size_t... I>
constexpr auto sum_impl(const Vec<T, N>& a, std::index_sequence<I...>);

template <typename T, std::size_t N, std::size_t... I>
constexpr auto product_impl(const Vec<T, N>& a, std::index_sequence<I...>);

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto hadamard_product_impl(const Vec<T1, N>& a, const Vec<T2, N>& b,
                                     std::index_sequence<I...>);

} // namespace detail

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

  constexpr Vec(const storage_type& storage) noexcept(std::is_nothrow_copy_constructible_v<storage_type>)
      : values(storage) {}
  constexpr Vec(storage_type&& storage) noexcept(std::is_nothrow_move_constructible_v<storage_type>)
      : values(std::move(storage)) {}

  template <typename... Args,
            typename = std::enable_if_t<sizeof...(Args) == N &&
                                        (std::is_convertible_v<Args, T> && ...)>>
  constexpr Vec(Args&&... args) noexcept((std::is_nothrow_constructible_v<T, Args&&> && ...))
      : values{static_cast<T>(std::forward<Args>(args))...} {}

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

  constexpr storage_type& storage() & noexcept { return values; }
  constexpr const storage_type& storage() const& noexcept { return values; }
  constexpr storage_type&& storage() && noexcept { return std::move(values); }
  constexpr const storage_type&& storage() const&& noexcept { return std::move(values); }
};

template <typename To, typename From, std::size_t N>
constexpr Vec<To, N>
array_cast(const Vec<From, N>& in)
{
  if constexpr (std::is_same_v<To, From>) {
    return in;
  } else {
    static_assert(detail::is_static_castable<To, From>::value,
                  "array_cast requires component-wise static_cast support");

    return detail::array_cast_impl<To>(in, std::make_index_sequence<N>{});
  }
}

template <typename To, typename From, std::size_t N>
constexpr Vec<To, N>
array_cast(const std::array<From, N>& in)
{
  return array_cast<To>(Vec<From, N>{in});
}

template <typename T, std::size_t N>
inline constexpr auto operator-(const Vec<T, N>& rhs) noexcept(noexcept(detail::unary_minus_impl(
    rhs, std::make_index_sequence<N>{}))) {
  return detail::unary_minus_impl(rhs, std::make_index_sequence<N>{});
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T1>>>
inline constexpr auto operator*(const T1& lhs, const Vec<T2, N>& rhs) noexcept(noexcept(
    detail::scalar_times_vec_impl(lhs, rhs, std::make_index_sequence<N>{}))) {
  return detail::scalar_times_vec_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T2>>>
inline constexpr auto operator*(const Vec<T1, N>& lhs, const T2& rhs) noexcept(noexcept(
    detail::vec_times_scalar_impl(lhs, rhs, std::make_index_sequence<N>{}))) {
  return detail::vec_times_scalar_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T2>>>
inline constexpr auto operator/(const Vec<T1, N>& lhs, const T2& rhs) noexcept(noexcept(
    detail::vec_div_scalar_impl(lhs, rhs, std::make_index_sequence<N>{}))) {
  return detail::vec_div_scalar_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T1>>>
inline constexpr auto operator/(const T1& lhs, const Vec<T2, N>& rhs) noexcept(noexcept(
    detail::scalar_div_vec_impl(lhs, rhs, std::make_index_sequence<N>{}))) {
  return detail::scalar_div_vec_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <typename T1, typename T2, std::size_t N>
inline constexpr auto operator+(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs) noexcept(noexcept(
    detail::vec_plus_vec_impl(lhs, rhs, std::make_index_sequence<N>{}))) {
  return detail::vec_plus_vec_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <typename T1, typename T2, std::size_t N>
inline constexpr auto operator-(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs) noexcept(noexcept(
    detail::vec_minus_vec_impl(lhs, rhs, std::make_index_sequence<N>{}))) {
  return detail::vec_minus_vec_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T2>>>
inline constexpr Vec<T1, N>& operator+=(Vec<T1, N>& lhs, const T2& rhs) noexcept(noexcept(
    std::declval<T1&>() += std::declval<const T2&>())) {
  for (auto& value : lhs) {
    value += rhs;
  }
  return lhs;
}

template <typename T1, typename T2, std::size_t N>
inline constexpr Vec<T1, N>& operator+=(Vec<T1, N>& lhs, const Vec<T2, N>& rhs) noexcept(noexcept(
    std::declval<T1&>() += std::declval<const T2&>())) {
  for (std::size_t i = 0; i < N; ++i) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T2>>>
inline constexpr Vec<T1, N>& operator-=(Vec<T1, N>& lhs, const T2& rhs) noexcept(noexcept(
    std::declval<T1&>() -= std::declval<const T2&>())) {
  for (auto& value : lhs) {
    value -= rhs;
  }
  return lhs;
}

template <typename T1, typename T2, std::size_t N>
inline constexpr Vec<T1, N>& operator-=(Vec<T1, N>& lhs, const Vec<T2, N>& rhs) noexcept(noexcept(
    std::declval<T1&>() -= std::declval<const T2&>())) {
  for (std::size_t i = 0; i < N; ++i) {
    lhs[i] -= rhs[i];
  }
  return lhs;
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T2>>>
inline constexpr Vec<T1, N>& operator*=(Vec<T1, N>& lhs, const T2& rhs) noexcept(noexcept(
    std::declval<T1&>() *= std::declval<const T2&>())) {
  for (auto& value : lhs) {
    value *= rhs;
  }
  return lhs;
}

template <typename T1, typename T2, std::size_t N,
          typename = std::enable_if_t<detail::is_container_scalar_v<T2>>>
inline constexpr Vec<T1, N>& operator/=(Vec<T1, N>& lhs, const T2& rhs) noexcept(noexcept(
    std::declval<T1&>() /= std::declval<const T2&>())) {
  for (auto& value : lhs) {
    value /= rhs;
  }
  return lhs;
}

/// Returns true if all components of the Vec are exactly equal, false otherwise.
template <typename T, std::size_t N>
inline constexpr bool equal(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept(noexcept(
    lhs.values == rhs.values)) {
  return lhs.values == rhs.values;
}

template <typename T, std::size_t N>
inline constexpr bool operator==(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept(noexcept(
    equal(lhs, rhs))) {
  return equal(lhs, rhs);
}

template <typename T, std::size_t N>
inline constexpr bool operator!=(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept(noexcept(
    lhs == rhs)) {
  return !(lhs == rhs);
}

template <typename T, std::size_t N>
inline constexpr bool operator<(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept(noexcept(
    lhs.values < rhs.values)) {
  return lhs.values < rhs.values;
}

template <typename T, std::size_t N>
inline constexpr bool operator>(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept(noexcept(
    rhs < lhs)) {
  return rhs < lhs;
}

template <typename T, std::size_t N>
inline constexpr bool operator<=(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept(noexcept(
    rhs < lhs)) {
  return !(rhs < lhs);
}

template <typename T, std::size_t N>
inline constexpr bool operator>=(const Vec<T, N>& lhs, const Vec<T, N>& rhs) noexcept(noexcept(
    lhs < rhs)) {
  return !(lhs < rhs);
}

template <typename T1, typename T2, std::size_t N>
inline constexpr auto operator%(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs) noexcept(noexcept(
    detail::vec_mod_vec_impl(lhs, rhs, std::make_index_sequence<N>{}))) {
  return detail::vec_mod_vec_impl(lhs, rhs, std::make_index_sequence<N>{});
}

/// Returns the dot product a . b
template <typename T1, typename T2, std::size_t N>
inline constexpr auto dot(const Vec<T1, N>& a, const Vec<T2, N>& b) noexcept(noexcept(
    detail::dot_impl(a, b, std::make_index_sequence<N>{}))) {
  return detail::dot_impl(a, b, std::make_index_sequence<N>{});
}

/// Returns the dot product of a and b, which is then squared.
template <typename T1, typename T2, std::size_t N>
inline constexpr auto dot_squared(const Vec<T1, N>& a, const Vec<T2, N>& b)
    noexcept(noexcept(dot(a, b) * dot(a, b))) -> decltype(dot(a, b) * dot(a, b)) {
  return dot(a, b) * dot(a, b);
}

/// Returns the Euclidean norm of the vector.
template <typename T, std::size_t N>
inline constexpr auto norm(const Vec<T, N>& a) -> decltype(std::sqrt(dot(a, a))) {
  return std::sqrt(dot(a, a));
}

/// Returns the square of the Euclidean norm of the vector.
template <typename T, std::size_t N>
inline constexpr auto norm_squared(const Vec<T, N>& a) noexcept(noexcept(dot(a, a)))
    -> decltype(dot(a, a)) {
  return dot(a, a);
}

/// Returns the sum of the elements in the vector a.
template <typename T, std::size_t N>
inline constexpr auto sum(const Vec<T, N>& a) noexcept(noexcept(
    detail::sum_impl(a, std::make_index_sequence<N>{}))) {
  return detail::sum_impl(a, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
inline constexpr auto sum(const std::array<T, N>& a) noexcept(noexcept(sum(Vec<T, N>{a}))) {
  return sum(Vec<T, N>{a});
}

/// Returns the product of the elements in the vector a.
template <typename T, std::size_t N>
inline constexpr auto product(const Vec<T, N>& a) noexcept(noexcept(
    detail::product_impl(a, std::make_index_sequence<N>{}))) {
  return detail::product_impl(a, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
inline constexpr auto product(const std::array<T, N>& a) noexcept(noexcept(product(Vec<T, N>{a}))) {
  return product(Vec<T, N>{a});
}

/// Returns a Vec with the element wise multiplication of a and b.
template <typename T1, typename T2, std::size_t N>
inline constexpr auto hadamard_product(const Vec<T1, N>& a, const Vec<T2, N>& b) noexcept(noexcept(
    detail::hadamard_product_impl(a, b, std::make_index_sequence<N>{}))) {
  return detail::hadamard_product_impl(a, b, std::make_index_sequence<N>{});
}

namespace detail {

template <typename To, typename From, std::size_t N, std::size_t... I>
constexpr Vec<To, N> array_cast_impl(const Vec<From, N>& in, std::index_sequence<I...>) {
  return {static_cast<To>(in[I])...};
}

template <typename T, std::size_t N, std::size_t... I>
constexpr auto unary_minus_impl(const Vec<T, N>& rhs, std::index_sequence<I...>) {
  return Vec<decltype(-rhs[0]), N>{(-rhs[I])...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto scalar_times_vec_impl(const T1& lhs, const Vec<T2, N>& rhs,
                                     std::index_sequence<I...>) {
  return Vec<decltype(lhs * rhs[0]), N>{(lhs * rhs[I])...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_times_scalar_impl(const Vec<T1, N>& lhs, const T2& rhs,
                                     std::index_sequence<I...>) {
  return Vec<decltype(lhs[0] * rhs), N>{(lhs[I] * rhs)...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_div_scalar_impl(const Vec<T1, N>& lhs, const T2& rhs,
                                   std::index_sequence<I...>) {
  return Vec<decltype(lhs[0] / rhs), N>{(lhs[I] / rhs)...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto scalar_div_vec_impl(const T1& lhs, const Vec<T2, N>& rhs,
                                   std::index_sequence<I...>) {
  return Vec<decltype(lhs / rhs[0]), N>{(lhs / rhs[I])...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_plus_vec_impl(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs,
                                 std::index_sequence<I...>) {
  return Vec<decltype(lhs[0] + rhs[0]), N>{(lhs[I] + rhs[I])...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_minus_vec_impl(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs,
                                  std::index_sequence<I...>) {
  return Vec<decltype(lhs[0] - rhs[0]), N>{(lhs[I] - rhs[I])...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto vec_mod_vec_impl(const Vec<T1, N>& lhs, const Vec<T2, N>& rhs,
                                std::index_sequence<I...>) {
  return Vec<decltype(lhs[0] % rhs[0]), N>{(lhs[I] % rhs[I])...};
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto dot_impl(const Vec<T1, N>& a, const Vec<T2, N>& b,
                        std::index_sequence<I...>) {
  return (... + (a[I] * b[I]));
}

template <typename T, std::size_t N, std::size_t... I>
constexpr auto sum_impl(const Vec<T, N>& a, std::index_sequence<I...>) {
  return (... + a[I]);
}

template <typename T, std::size_t N, std::size_t... I>
constexpr auto product_impl(const Vec<T, N>& a, std::index_sequence<I...>) {
  return (... * a[I]);
}

template <typename T1, typename T2, std::size_t N, std::size_t... I>
constexpr auto hadamard_product_impl(const Vec<T1, N>& a, const Vec<T2, N>& b,
                                     std::index_sequence<I...>) {
  return Vec<decltype(a[0] * b[0]), N>{(a[I] * b[I])...};
}

} // namespace detail

static_assert(sizeof(Vec<double, 3>) == sizeof(double) * 3,
              "Vec must not add storage padding");
static_assert(std::is_trivially_copyable_v<Vec<double, 3>>,
              "Vec must be trivially copyable");
static_assert(std::is_standard_layout_v<Vec<double, 3>>,
              "Vec must use standard layout storage");

} // namespace jams

#endif // JAMS_VEC_H
