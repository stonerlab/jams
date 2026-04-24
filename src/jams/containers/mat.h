//
// Created by Joe Barker on 2017/08/26.
//

#ifndef JAMS_MAT_H
#define JAMS_MAT_H

#include <array>
#include <complex>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "jams/containers/vec.h"
#include "jams/helpers/mixed_precision.h"

namespace jams {

template <typename T, std::size_t M, std::size_t N>
struct Mat {
  static_assert(M > 0 && N > 0, "Mat requires at least one row and column");

  using value_type = T;
  using size_type = std::size_t;
  using row_type = std::array<T, M>;
  using storage_type = std::array<row_type, N>;
  using reference = row_type&;
  using const_reference = const row_type&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;

  storage_type values{};

  constexpr Mat() = default;
  constexpr Mat(const Mat&) = default;
  constexpr Mat(Mat&&) = default;
  constexpr Mat& operator=(const Mat&) = default;
  constexpr Mat& operator=(Mat&&) = default;
  ~Mat() = default;

  constexpr Mat(const storage_type& storage) : values(storage) {}
  constexpr Mat(storage_type&& storage) : values(std::move(storage)) {}

  template <typename... Args,
            typename = std::enable_if_t<sizeof...(Args) == M * N &&
                                        (std::is_convertible_v<Args, T> && ...)>>
  constexpr Mat(Args&&... args)
      : values(make_storage(std::forward_as_tuple(std::forward<Args>(args)...),
                            std::make_index_sequence<N>{})) {}

  constexpr reference operator[](size_type i) noexcept { return values[i]; }
  constexpr const_reference operator[](size_type i) const noexcept { return values[i]; }

  constexpr pointer data() noexcept { return values[0].data(); }
  constexpr const_pointer data() const noexcept { return values[0].data(); }

  constexpr iterator begin() noexcept { return values.begin(); }
  constexpr const_iterator begin() const noexcept { return values.begin(); }
  constexpr const_iterator cbegin() const noexcept { return values.cbegin(); }

  constexpr iterator end() noexcept { return values.end(); }
  constexpr const_iterator end() const noexcept { return values.end(); }
  constexpr const_iterator cend() const noexcept { return values.cend(); }

  static constexpr size_type rows() noexcept { return N; }
  static constexpr size_type cols() noexcept { return M; }
  static constexpr size_type size() noexcept { return N; }
  static constexpr bool empty() noexcept { return false; }

  constexpr operator storage_type&() noexcept { return values; }
  constexpr operator const storage_type&() const noexcept { return values; }

 private:
  template <std::size_t Row, typename Tuple, std::size_t... Cols>
  static constexpr row_type make_row(Tuple&& args, std::index_sequence<Cols...>) {
    return {static_cast<T>(std::get<Row * M + Cols>(std::forward<Tuple>(args)))...};
  }

  template <typename Tuple, std::size_t... Rows>
  static constexpr storage_type make_storage(Tuple&& args, std::index_sequence<Rows...>) {
    return {make_row<Rows>(std::forward<Tuple>(args), std::make_index_sequence<M>{})...};
  }
};

template <typename>
struct is_mat : std::false_type {};

template <typename T, std::size_t M, std::size_t N>
struct is_mat<Mat<T, M, N>> : std::true_type {};

template <typename To, typename From, std::size_t M, std::size_t N>
constexpr Mat<To, M, N>
matrix_cast(const Mat<From, M, N>& in)
{
  if constexpr (std::is_same_v<To, From>) {
    return in;
  } else {
    static_assert(std::is_arithmetic_v<To>,
                  "matrix_cast requires arithmetic To type");
    static_assert(std::is_arithmetic_v<From>,
                  "matrix_cast requires arithmetic From type");

    Mat<To, M, N> out{};
    for (std::size_t row = 0; row < N; ++row) {
      for (std::size_t col = 0; col < M; ++col) {
        out[row][col] = static_cast<To>(in[row][col]);
      }
    }
    return out;
  }
}

template <typename To, typename From, std::size_t M, std::size_t N>
constexpr Mat<To, M, N>
matrix_cast(const std::array<std::array<From, M>, N>& in)
{
  return matrix_cast<To>(Mat<From, M, N>{in});
}

template <typename T, std::size_t N>
constexpr Mat<T, N, N> identity()
{
  Mat<T, N, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    out[i][i] = T{1};
  }
  return out;
}

template <typename T1, typename T2, std::size_t M, std::size_t N>
inline constexpr auto operator*(const Mat<T1, M, N>& lhs, const Vec<T2, M>& rhs)
    -> Vec<decltype(lhs[0][0] * rhs[0]), N> {
  Vec<decltype(lhs[0][0] * rhs[0]), N> result{};
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t col = 0; col < M; ++col) {
      result[row] += lhs[row][col] * rhs[col];
    }
  }
  return result;
}

template <typename T1, typename T2, std::size_t M, std::size_t N,
          typename = std::enable_if_t<!is_mat<std::decay_t<T1>>::value>>
inline constexpr auto operator*(const T1& lhs, const Mat<T2, M, N>& rhs)
    -> Mat<decltype(lhs * rhs[0][0]), M, N> {
  Mat<decltype(lhs * rhs[0][0]), M, N> result{};
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t col = 0; col < M; ++col) {
      result[row][col] = lhs * rhs[row][col];
    }
  }
  return result;
}

template <typename T1, typename T2, std::size_t M, std::size_t N>
inline constexpr auto operator/(const Mat<T1, M, N>& lhs, const T2& rhs)
    -> Mat<decltype(lhs[0][0] / rhs), M, N> {
  Mat<decltype(lhs[0][0] / rhs), M, N> result{};
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t col = 0; col < M; ++col) {
      result[row][col] = lhs[row][col] / rhs;
    }
  }
  return result;
}

template <typename T1, typename T2, std::size_t M, std::size_t N>
inline constexpr auto operator+(const Mat<T1, M, N>& lhs, const Mat<T2, M, N>& rhs)
    -> Mat<decltype(lhs[0][0] + rhs[0][0]), M, N> {
  Mat<decltype(lhs[0][0] + rhs[0][0]), M, N> result{};
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t col = 0; col < M; ++col) {
      result[row][col] = lhs[row][col] + rhs[row][col];
    }
  }
  return result;
}

template <typename T1, typename T2, std::size_t M, std::size_t N>
inline constexpr auto operator-(const Mat<T1, M, N>& lhs, const Mat<T2, M, N>& rhs)
    -> Mat<decltype(lhs[0][0] - rhs[0][0]), M, N> {
  Mat<decltype(lhs[0][0] - rhs[0][0]), M, N> result{};
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t col = 0; col < M; ++col) {
      result[row][col] = lhs[row][col] - rhs[row][col];
    }
  }
  return result;
}

template <typename T, std::size_t M, std::size_t N>
inline constexpr auto operator-(const Mat<T, M, N>& a) -> Mat<decltype(-a[0][0]), M, N> {
  Mat<decltype(-a[0][0]), M, N> result{};
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t col = 0; col < M; ++col) {
      result[row][col] = -a[row][col];
    }
  }
  return result;
}

template <typename T1, typename T2, std::size_t M, std::size_t N, std::size_t P>
inline constexpr auto operator*(const Mat<T1, M, N>& lhs, const Mat<T2, P, M>& rhs)
    -> Mat<decltype(lhs[0][0] * rhs[0][0]), P, N> {
  Mat<decltype(lhs[0][0] * rhs[0][0]), P, N> result{};
  for (std::size_t row = 0; row < N; ++row) {
    for (std::size_t col = 0; col < P; ++col) {
      for (std::size_t k = 0; k < M; ++k) {
        result[row][col] += lhs[row][k] * rhs[k][col];
      }
    }
  }
  return result;
}

static_assert(sizeof(Mat<double, 3, 3>) == sizeof(double) * 9,
              "Mat must not add storage padding");
static_assert(std::is_trivially_copyable_v<Mat<double, 3, 3>>,
              "Mat must be trivially copyable");
static_assert(std::is_standard_layout_v<Mat<double, 3, 3>>,
              "Mat must use standard layout storage");

template <typename T, std::size_t M, std::size_t N>
constexpr bool operator==(const Mat<T, M, N>& lhs, const Mat<T, M, N>& rhs) {
  return lhs.values == rhs.values;
}

template <typename T, std::size_t M, std::size_t N>
constexpr bool operator!=(const Mat<T, M, N>& lhs, const Mat<T, M, N>& rhs) {
  return !(lhs == rhs);
}

} // namespace jams

template <typename T, std::size_t M, std::size_t N>
using Mat = jams::Mat<T, M, N>;

using jams::identity;
using jams::matrix_cast;

#endif // JAMS_MAT_H
