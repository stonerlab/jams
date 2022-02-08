// cuda_only_implementation_macro.h                                    -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_ONLY_IMPLEMENTATION_MACRO
#define INCLUDED_JAMS_CUDA_ONLY_IMPLEMENTATION_MACRO

#include <stdexcept>

#if HAS_CUDA
#define CUDA_ONLY_IMPLEMENTATION(function) function;
#else
#define CUDA_ONLY_IMPLEMENTATION(function) inline function {throw std::runtime_error(#function " not implemented for CPU only build");}
#endif

#endif
// ----------------------------- END-OF-FILE ----------------------------------