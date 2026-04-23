#ifndef JAMS_MULTIARRAY_NUMERIC_H
#define JAMS_MULTIARRAY_NUMERIC_H

#include <algorithm>
#include <array>
#include <complex>
#include <type_traits>

namespace jams {
namespace detail {

template<typename T>
struct supports_fast_zero : std::bool_constant<std::is_integral_v<T> ||
                                                std::is_floating_point_v<T> ||
                                                std::is_enum_v<T>> {};

template<typename T>
struct supports_fast_zero<std::complex<T>> : supports_fast_zero<T> {};

template<typename T, std::size_t N>
struct supports_fast_zero<std::array<T, N>> : supports_fast_zero<T> {};

template<typename T>
inline constexpr bool supports_fast_zero_v = supports_fast_zero<T>::value;

}  // namespace detail

template<class FTp_, std::size_t FDim_, class FIdx_>
inline void zero(MultiArray<FTp_, FDim_, FIdx_>& x) {
  if constexpr (detail::supports_fast_zero_v<FTp_>) {
    x.data_.zero();
  } else {
    std::fill(x.begin(), x.end(), FTp_{});
  }
}

template<class FTp_, std::size_t FDim_, class FIdx_, class Tp2_>
inline void element_scale(MultiArray<FTp_, FDim_, FIdx_>& x, const Tp2_& y) {
  std::transform(x.begin(), x.end(), x.begin(), [y](const FTp_ &a) { return a * y; });
}

template<class FTp_, std::size_t FDim_, class FIdx_>
inline void element_sum(MultiArray<FTp_, FDim_, FIdx_>& x, const MultiArray<FTp_, FDim_, FIdx_>& y) {
  assert(x.shape() == y.shape());
  std::transform(y.begin(), y.end(), x.begin(), x.begin(),
                 [](const FTp_&x, const FTp_ &y) -> FTp_ { return x + y; });
}

}  // namespace jams

#endif  // JAMS_MULTIARRAY_NUMERIC_H
