// array_ops.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_ARRAY_OPS
#define INCLUDED_JAMS_ARRAY_OPS

#include <jams/containers/multiarray.h>
#include <jams/containers/vec3.h>

namespace jams {
template<class T>
inline std::array<T, 3> reduce_vector_field(const jams::MultiArray<T, 2> &x) {
  assert(x.size(1) == 3);
  std::array<T, 3> sum = {0, 0, 0};
  for (auto i = 0; i < x.size(0); ++i) {
    for (auto j = 0; j < 3; ++j) {
      sum[j] += x(i, j);
    }
  }
  return sum;
}
}


#endif
// ----------------------------- END-OF-FILE ----------------------------------