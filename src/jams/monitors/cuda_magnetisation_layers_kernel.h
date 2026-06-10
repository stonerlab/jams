// cuda_magnetisation_layers_kernel.h                                  -*-C++-*-
#ifndef INCLUDED_JAMS_MONITORS_CUDA_MAGNETISATION_LAYERS_KERNEL
#define INCLUDED_JAMS_MONITORS_CUDA_MAGNETISATION_LAYERS_KERNEL

#include <jams/core/types.h>
#include <jams/cuda/cuda_only_implementation_macro.h>

#if HAS_CUDA
class CudaStream;

CUDA_ONLY_IMPLEMENTATION(
void execute_cuda_magnetisation_layers_kernel(
    CudaStream& stream,
    int num_layers,
    int num_chunks,
    const int* chunk_begin_offsets,
    const int* chunk_end_offsets,
    const int* chunk_layer_indices,
    const int* spin_indices,
    const double* spins,
    const jams::Real* moments,
    double* layer_magnetisation));
#endif

#endif
// ----------------------------- END-OF-FILE ----------------------------------
