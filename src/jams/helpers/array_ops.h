// array_ops.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_ARRAY_OPS
#define INCLUDED_JAMS_ARRAY_OPS

#include <jams/containers/multiarray.h>
#include <jams/containers/vec3.h>

namespace jams {
template<class T>
inline std::array<T, 3> vector_field_reduce(const jams::MultiArray<T, 2> &x) {
  assert(x.size(1) == 3);

  // Kahan sum over the field components to keep precision for long
  // vectors.

  std::array<T, 3> sum = {x(0,0), x(0,1), x(0,2)};
  std::array<T, 3> c = {0, 0, 0};

  for (auto i = 1; i < x.size(0); ++i) {
    std::array<T, 3> y = {
        x(i, 0) - c[0],
        x(i, 1) - c[1],
        x(i, 2) - c[2]};
    std::array<T, 3> t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

template<class T>
inline std::array<T, 3> vector_field_indexed_reduce(const jams::MultiArray<T, 2> &x, const jams::MultiArray<int, 1>& indices) {
  assert(x.size(1) == 3);

  // Kahan sum over the field components to keep precision for long
  // vectors.

  std::array<T, 3> sum = {x(indices(0),0), x(indices(0),1), x(indices(0),2)};
  std::array<T, 3> c = {0, 0, 0};

  for (auto i = 1; i < indices.size(); ++i) {
    std::array<T, 3> y = {
        x(indices(i), 0) - c[0],
        x(indices(i), 1) - c[1],
        x(indices(i), 2) - c[2]};
    std::array<T, 3> t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

template<class T>
inline T scalar_field_indexed_reduce(const jams::MultiArray<T, 1> &x, const jams::MultiArray<int, 1>& indices) {
  // Kahan sum over the field components to keep precision for long
  // vectors.

  T sum = x(indices(0));
  T c = 0;

  for (auto i = 1; i < indices.size(); ++i) {
    T y = x(indices(i)) - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}
}


#endif
// ----------------------------- END-OF-FILE ----------------------------------