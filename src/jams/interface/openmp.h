//
// Created by Joseph Barker on 2019-03-31.
//

#ifndef JAMS_OPENMP_H
#define JAMS_OPENMP_H

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#if HAS_OMP
  #define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
  #define OMP_PARALLEL_FOR
#endif

#endif //JAMS_OPENMP_H
