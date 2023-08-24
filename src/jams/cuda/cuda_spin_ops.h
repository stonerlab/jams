// cuda_spin_ops.h                                                     -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_SPIN_OPS
#define INCLUDED_JAMS_CUDA_SPIN_OPS

#include <jams/containers/multiarray.h>
#include <jams/containers/mat3.h>
#include <jams/cuda/cuda_only_implementation_macro.h>

namespace jams {

/// Normalise spins to unit vectors
CUDA_ONLY_IMPLEMENTATION(
    void normalise_spins_cuda(jams::MultiArray<double, 2> &spins));

/// Rotate spins with given indices by the rotation matrix
///
/// @warning No check is made that rotation_matrix is unitary.
CUDA_ONLY_IMPLEMENTATION(
    void rotate_spins_cuda(jams::MultiArray<double, 2> &spins,
                           const Mat3 &rotation_matrix,
                           const jams::MultiArray<int, 1> &indices));


/// Scale spins by given factor
///
/// @warning This scales the spin vectors to be non-unit vectors. This only
/// makes sense for solves which do not assume the spins to be unit vectors.
CUDA_ONLY_IMPLEMENTATION(
    void scale_spins_cuda(jams::MultiArray<double, 2> &spins,
                          const double &scale_factor,
                          const jams::MultiArray<int, 1> &indices));


CUDA_ONLY_IMPLEMENTATION(
    void add_to_spin_length_cuda(jams::MultiArray<double, 2> &spins,
                                 const double &additional_length,
                                 const jams::MultiArray<int, 1> &indices));

CUDA_ONLY_IMPLEMENTATION(
    void add_to_spin_cuda(jams::MultiArray<double, 2> &spins,
                          const Vec3 &additional_length,
                          const jams::MultiArray<int, 1> &indices));

}

#endif
// ----------------------------- END-OF-FILE ----------------------------------