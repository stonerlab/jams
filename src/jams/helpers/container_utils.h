//
// Created by Joseph Barker on 06/10/2025.
//

#ifndef JAMS_CONTAINER_UTILS_H
#define JAMS_CONTAINER_UTILS_H
#include <type_traits>
#include <algorithm>
#include <utility>

namespace jams {
// Generic "zero-like" factory
template<typename T>
constexpr T zero_like() {
  if constexpr (std::is_arithmetic_v<T>) return T{0};
  else if constexpr (std::is_enum_v<T>) return static_cast<T>(0);
  else return T{};  // default constructed
}

// Zero-fill a single container
template<typename Container>
void zero_container(Container& c) {
  using value_type = typename std::decay_t<decltype(*std::begin(c))>;
  std::fill(std::begin(c), std::end(c), zero_like<value_type>());
}

// Variadic version — zero multiple containers at once
template<typename... Containers>
void zero_all(Containers&... containers) {
  (zero_container(containers), ...);
}

}
#endif //JAMS_CONTAINER_UTILS_H