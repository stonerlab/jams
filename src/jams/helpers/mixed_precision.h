//
// Created by Joseph Barker on 11/12/2025.
//

#ifndef JAMS_MIXED_PRECISION_H
#define JAMS_MIXED_PRECISION_H

#include <complex>

#if HAS_CUDA
#include <cufft.h>
#endif

namespace jams {
#if DO_MIXED_PRECISION

using RealHi = double;
using Real = float;

using ComplexHi = std::complex<double>;
using ComplexLo = std::complex<float>;

#if HAS_CUDA
// TODO I think this is binary compatible with std::complex, so maybe not needed
using cufftComplexLo = cufftComplex;
using Real2 = float2;
using Real3 = float3;
#endif


#else
using RealHi = double;
using Real = double;

using ComplexHi = std::complex<double>;
using ComplexLo = std::complex<double>;
#if HAS_CUDA
using cufftComplexLo = cufftDoubleComplex;
using Real2 = double2;
using Real3 = double3;
#endif
#endif
}

#endif //JAMS_MIXED_PRECISION_H