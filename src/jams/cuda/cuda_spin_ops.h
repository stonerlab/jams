// cuda_spin_ops.h                                                     -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_SPIN_OPS
#define INCLUDED_JAMS_CUDA_SPIN_OPS

#include <jams/containers/multiarray.h>
#include <jams/containers/mat3.h>
#include <jams/cuda/cuda_only_implementation_macro.h>

namespace jams {

/// Rotate spins with given indices by the rotation matrix
///
/// @warning No check is made that rotation_matrix is unitary.
CUDA_ONLY_IMPLEMENTATION(
    void rotate_spins_cuda(jams::MultiArray<double, 2> &spins,
                           const Mat3 &rotation_matrix,
                           const jams::MultiArray<int, 1> &indices));
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------