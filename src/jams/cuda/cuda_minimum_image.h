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
#include <jams/helpers/mixed_precision.h>

namespace jams {
  void cuda_minimum_image(const jams::Vec<jams::Real, 3> &a, const jams::Vec<jams::Real, 3> &b, const jams::Vec<jams::Real, 3> &c, const jams::Vec<bool, 3> &pbc,
                          const jams::Vec<jams::Real, 3> &r_i, const jams::MultiArray<jams::Real,2>& r, jams::MultiArray<jams::Real,2>& r_ij);
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------
