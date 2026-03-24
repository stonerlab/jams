#ifndef JAMS_CUDA_FUNCTIONAL_H
#define JAMS_CUDA_FUNCTIONAL_H

#include <thrust/functional.h>

#if defined(__has_include)
#if __has_include(<cuda/std/functional>)
#include <cuda/std/functional>
#define JAMS_HAS_CUDA_STD_FUNCTIONAL 1
#endif
#endif

namespace jams {
namespace cuda_compat {

#if defined(JAMS_HAS_CUDA_STD_FUNCTIONAL)
template<typename T>
using plus = cuda::std::plus<T>;
#else
template<typename T>
using plus = thrust::plus<T>;
#endif

}  // namespace cuda_compat
}  // namespace jams

#endif  // JAMS_CUDA_FUNCTIONAL_H
