//
// Created by Codex on 2026-02-16.
//

#ifndef JAMS_HKL_INDEX_H
#define JAMS_HKL_INDEX_H

#include <jams/core/types.h>
#include <jams/helpers/defaults.h>
#include <jams/interface/fft.h>

namespace jams {

/// @brief Reciprocal-space point with FFT index metadata.
struct HKLIndex {
  Vec<double, 3> hkl;          ///< Reciprocal lattice point in fractional units.
  Vec<double, 3> xyz;          ///< Reciprocal lattice point in cartesian units.
  FFTWHermitianIndex<3> index; ///< FFTW 3D array index and conjugation flag.
};

inline bool operator==(const HKLIndex& a, const HKLIndex& b)
{
  return jams::approximately_equal(a.hkl, b.hkl, jams::defaults::lattice_tolerance);
}

} // namespace jams

#endif // JAMS_HKL_INDEX_H
