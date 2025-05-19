#ifndef JAMS_CORE_BLAS_H
#define JAMS_CORE_BLAS_H

#if defined(HAS_MKL)
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#endif
