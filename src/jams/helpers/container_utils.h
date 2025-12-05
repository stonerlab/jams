//
// Created by Joseph Barker on 05/12/2025.
//

#ifndef JAMS_CONTAINER_UTILS_H
#define JAMS_CONTAINER_UTILS_H

#include <vector>

template <typename T>
std::vector<T> make_reserved(std::size_t n) {
  std::vector<T> v;
  v.reserve(n);
  return v;
}

#endif //JAMS_CONTAINER_UTILS_H