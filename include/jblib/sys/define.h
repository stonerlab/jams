#ifndef JBLIB_SYS_DEFINE_H
#define JBLIB_SYS_DEFINE_H

#define JBLIB_VERSION 0.3.4
// GCC attributes

//#define JBLIB_ALIGNED

#ifdef JBLIB_ALIGNED
#if(defined __GNUG__ || defined __clang__)  // GCC
#define aligned16     __attribute__((aligned(16)))
#define aligned64     __attribute__((aligned(64)))
// #warning "Host aligned arrays currently break CUDA"
#elif (defined __CUDACC__) // NVCC
#define aligned16     __align__(16)
#define aligned64     __align__(64)
#else
#error "No alignment attribute for your compiler"
#endif
#else
#define aligned16
#define aligned64
#endif

#define force_inline __attribute__((always_inline))

#define restrict     __restrict__

#endif  // JBLIB_SYS_DEFINE_H
