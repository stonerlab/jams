// spinops.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_SPINOPS
#define INCLUDED_JAMS_SPINOPS
/// @brief:
///
/// @details: This component...
///
/// Usage
/// -----

#include <jams/containers/multiarray.h>

namespace jams {

/// Rotate all spins by the rotation matrix
///
/// @warning No check is made that rotation_matrix is unitary.
void
rotate_spins(jams::MultiArray<double, 2> &spins, const Mat3 &rotation_matrix);


/// Rotate spins with given indices by the rotation matrix
///
/// @warning No check is made that rotation_matrix is unitary.
void
rotate_spins(jams::MultiArray<double, 2> &spins, const Mat3 &rotation_matrix,
             const jams::MultiArray<int, 1> &indices);


/// Returns the sum of the spins with the given indices
Vec3 sum_spins(
    const jams::MultiArray<double, 2>& spins,
    const jams::MultiArray<int, 1>& indices);


/// Returns the sum of all spins multiplied by their moments
Vec3 sum_spins_moments(
    const jams::MultiArray<double, 2>& spins,
    const jams::MultiArray<double, 1>& moments);


/// Returns the sum of the spins multiplied by their moments with the given indices
Vec3 sum_spins_moments(
    const jams::MultiArray<double, 2>& spins,
    const jams::MultiArray<double, 1>& moments,
    const jams::MultiArray<int, 1>& indices);



}

#endif
// ----------------------------- END-OF-FILE ----------------------------------