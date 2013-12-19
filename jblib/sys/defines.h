#ifndef SYS_DEFINES_H
#define SYS_DEFINES_H

#define JBLIB_VERSION 0.2
// GCC attributes

#define ALIGNTYPE16     __attribute__((aligned(16)))
#define ALIGNTYPE64     __attribute__((aligned(64)))

//#if __has_builtin(__builtin_assume_aligned)
#if(!(defined __llvm__) && defined __GNUG__)
    #define JB_ASSUME_ALIGNED16( x ) __builtin_assume_aligned(x,16)
    #define JB_ASSUME_ALIGNED64( x ) __builtin_assume_aligned(x,64)
#else
    #define JB_ASSUME_ALIGNED16( x ) x
    #define JB_ASSUME_ALIGNED64( x ) x
#endif

#define JB_INLINE       inline
#define JB_FORCE_INLINE __attribute__((always_inline)) inline

#define JB_RESTRICT     __restrict__

#define JB_REGISTER     register

#endif
