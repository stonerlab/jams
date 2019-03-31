//
// Created by Joseph Barker on 2019-03-31.
//

#ifndef JAMS_OPENMP_H
#define JAMS_OPENMP_H

#define DO_PRAGMA(x) _Pragma (#x)

#if HAS_OMP
  #define OMP(x) \
    DO_PRAGMA("omp" x)

  #define OMP_PARALLEL_FOR \
    DO_PRAGMA("omp parallel for schedule(static)")
#else
  #define OMP(x)

  #define OMP_PARALLEL_FOR
#endif

#endif //JAMS_OPENMP_H
