// cuda_only_implementation_macro.h                                    -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_ONLY_IMPLEMENTATION_MACRO
#define INCLUDED_JAMS_CUDA_ONLY_IMPLEMENTATION_MACRO

#include <stdexcept>

// ... existing code ...
#include <stdexcept>

#if HAS_CUDA
#define CUDA_ONLY_IMPLEMENTATION(...) __VA_ARGS__;
#else
#define CUDA_ONLY_IMPLEMENTATION(...) inline __VA_ARGS__ {throw std::runtime_error(#__VA_ARGS__ " not implemented for CPU only build");}
#endif

#endif
// ... existing code ...
// ----------------------------- END-OF-FILE ----------------------------------
