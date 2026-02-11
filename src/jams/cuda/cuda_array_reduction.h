// cuda_array_reduction.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_ARRAY_REDUCTION
#define INCLUDED_JAMS_CUDA_ARRAY_REDUCTION

#include <jams/containers/vec3.h>
#include <jams/containers/multiarray.h>
#include <jams/cuda/cuda_only_implementation_macro.h>

namespace jams {

/// Reduce a vector field (N x 3 MultiArray) on the GPU
    CUDA_ONLY_IMPLEMENTATION(
            Vec3 vector_field_reduce_cuda(const jams::MultiArray<double, 2> &x));


    CUDA_ONLY_IMPLEMENTATION(
            Vec3 vector_field_indexed_reduce_cuda(const jams::MultiArray<double, 2> &x,
                                                  const jams::MultiArray<int, 1> &indices));


    CUDA_ONLY_IMPLEMENTATION(
            Vec3 vector_field_scale_and_reduce_cuda(const jams::MultiArray<double, 2> &x,
                                                    const jams::MultiArray<double, 1> &scale_factors));

    #if HAS_CUDA
    CUDA_ONLY_IMPLEMENTATION(
            template<typename X, typename S>
            std::array<X,3> vector_field_indexed_scale_and_reduce_cuda(const jams::MultiArray<X, 2> &x,
                                                            const jams::MultiArray<S, 1> &scale_factors,
                                                            const jams::MultiArray<int, 1> &indices));
    #else
    template<typename X, typename S>
    inline std::array<X,3> vector_field_indexed_scale_and_reduce_cuda(const jams::MultiArray<X, 2> &x,
                                                            const jams::MultiArray<S, 1> &scale_factors,
                                                            const jams::MultiArray<int, 1> &indices) {
        throw std::runtime_error("vector_field_indexed_scale_and_reduce_cuda not implemented for CPU only build");
    }
    #endif
#if HAS_CUDA
        CUDA_ONLY_IMPLEMENTATION(
                float scalar_field_reduce_cuda(const jams::MultiArray<float, 1> &x, cudaStream_t stream = nullptr));

        CUDA_ONLY_IMPLEMENTATION(
                double scalar_field_reduce_cuda(const jams::MultiArray<double, 1> &x, cudaStream_t stream = nullptr));
#endif
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------
