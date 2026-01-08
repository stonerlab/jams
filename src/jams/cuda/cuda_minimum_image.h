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
  void cuda_minimum_image(const Vec3R &a, const Vec3R &b, const Vec3R &c, const Vec3b &pbc,
                          const Vec3R &r_i, const jams::MultiArray<jams::Real,2>& r, jams::MultiArray<jams::Real,2>& r_ij);
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------