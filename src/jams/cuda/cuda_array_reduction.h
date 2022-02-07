// cuda_array_reduction.h                                              -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_ARRAY_REDUCTION
#define INCLUDED_JAMS_CUDA_ARRAY_REDUCTION

#include <jams/containers/vec3.h>
#include <jams/containers/multiarray.h>

namespace jams {

/// Reduce a vector field (N x 3 MultiArray) on the GPU
Vec3 vector_field_reduce_cuda(const jams::MultiArray<double, 2> &x);

Vec3 vector_field_indexed_reduce_cuda(const jams::MultiArray<double, 2>& x, const jams::MultiArray<int, 1>& indices);

Vec3 vector_field_scale_and_reduce_cuda(const jams::MultiArray<double, 2> &x, const jams::MultiArray<double, 1>& scale_factors);

Vec3 vector_field_indexed_scale_and_reduce_cuda(const jams::MultiArray<double, 2> &x, const jams::MultiArray<double, 1>& scale_factors, const jams::MultiArray<int, 1>& indices);
}

#endif
// ----------------------------- END-OF-FILE ----------------------------------