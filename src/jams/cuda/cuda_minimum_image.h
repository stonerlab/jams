// cuda_minimum_image.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_MINIMUM_IMAGE
#define INCLUDED_JAMS_CUDA_MINIMUM_IMAGE
/// @brief:
///
/// @details: This component...
///
/// Usage
/// -----

#include <jams/containers/vec3.h>
#include <jams/containers/multiarray.h>

namespace jams {
  void cuda_minimum_image(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3b &pbc,
                          const Vec3 &r_i, const jams::MultiArray<double,2>& r, jams::MultiArray<double,2>& r_ij);
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------