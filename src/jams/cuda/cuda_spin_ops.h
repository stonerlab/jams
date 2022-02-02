// cuda_spin_ops.h                                                     -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_SPIN_OPS
#define INCLUDED_JAMS_CUDA_SPIN_OPS

#include <jams/containers/multiarray.h>
#include <jams/containers/vec3.h>

namespace jams {

/// Returns the sum of the spins with the given indices
Vec3 cuda_sum_spins(
    const jams::MultiArray<double, 2>& spins,
    const jams::MultiArray<int, 1>& indices);


/// Rotate spins with given indices by the rotation matrix
///
/// @warning No check is made that rotation_matrix is unitary.
void
cuda_rotate_spins(jams::MultiArray<double, 2> &spins, const Mat3 &rotation_matrix,
                  const jams::MultiArray<int, 1> &indices);

/// Returns the sum of the spins multiplied by their moments with the given indices
Vec3 cuda_sum_spins_moments(
    const jams::MultiArray<double, 2>& spins,
    const jams::MultiArray<double, 1>& moments,
    const jams::MultiArray<int, 1>& indices);
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------