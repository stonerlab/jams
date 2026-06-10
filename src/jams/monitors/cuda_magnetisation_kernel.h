// cuda_magnetisation_kernel.h                                        -*-C++-*-
#ifndef INCLUDED_JAMS_MONITORS_CUDA_MAGNETISATION_KERNEL
#define INCLUDED_JAMS_MONITORS_CUDA_MAGNETISATION_KERNEL

#include <jams/core/types.h>
#include <jams/cuda/cuda_only_implementation_macro.h>

#if HAS_CUDA
class CudaStream;

CUDA_ONLY_IMPLEMENTATION(
void execute_cuda_magnetisation_kernel(
    CudaStream& stream,
    int num_groups,
    int num_chunks,
    const int* chunk_begin_offsets,
    const int* chunk_end_offsets,
    const int* chunk_group_indices,
    const int* spin_indices,
    const double* spins,
    const jams::Real* moments,
    double* group_magnetisation));
#endif

#endif
// ----------------------------- END-OF-FILE ----------------------------------
