//
// Created by Joseph Barker on 11/12/2025.
//

#ifndef JAMS_MIXED_PRECISION_H
#define JAMS_MIXED_PRECISION_H

#include <complex>

#if HAS_CUDA
#include <cufft.h>
#endif

#if DO_MIXED_PRECISION
        using ComplexLo = std::complex<float>;

        #if HAS_CUDA
        using cufftComplexLo = cufftComplex;
        #endif
#else
        using ComplexLo = std::complex<double>;
        #if HAS_CUDA
        using cufftComplexLo = cufftDoubleComplex;
        #endif
#endif

#endif //JAMS_MIXED_PRECISION_H