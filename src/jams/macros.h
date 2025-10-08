//
// Created by Joseph Barker on 2019-04-04.
//

#ifndef JAMS_MACROS_H
#define JAMS_MACROS_H

#ifdef NO_ALIGN
  #define ALIGNED(x)
#else
  #if(defined __GNUG__ || defined __clang__)  // GCC
    #define ALIGNED(x)     __attribute__((aligned(x)))
  #elif (defined __CUDACC__) // NVCC
    #define ALIGNED(x)     __align__(x)
  #else
    #error "No alignment attribute for your compiler"
  #endif
#endif

#ifdef _MSC_VER
#define JAMS_ALWAYS_INLINE _forceinline
#elif __GNUC__
#define JAMS_ALWAYS_INLINE __attribute__((always_inline))
#else
#define JAMS_ALWAYS_INLINE inline
#endif

#define RESTRICT     __restrict__

#if defined(__GNUC__) || defined(__clang__)
#  define UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#  define UNREACHABLE() __assume(false)
#else
#  define UNREACHABLE() ((void)0)
#endif

#endif //JAMS_MACROS_H
