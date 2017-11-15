#ifndef JAMS_CORE_BLAS_H
#define JAMS_CORE_BLAS_H

#if defined(USE_MKL)
#include <mkl_cblas.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#endif
