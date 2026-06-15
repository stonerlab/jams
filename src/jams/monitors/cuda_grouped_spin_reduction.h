// cuda_grouped_spin_reduction.h                                      -*-C++-*-
#ifndef INCLUDED_JAMS_MONITORS_CUDA_GROUPED_SPIN_REDUCTION
#define INCLUDED_JAMS_MONITORS_CUDA_GROUPED_SPIN_REDUCTION

#include <jams/containers/multiarray.h>
#include <jams/core/types.h>
#include <jams/cuda/cuda_only_implementation_macro.h>
#include <jams/monitors/spin_grouping.h>

#include <cstddef>
#include <vector>

class CudaStream;
class Lattice;

#if HAS_CUDA
namespace jams::monitors {

inline constexpr std::size_t kCudaGroupedSpinReductionChunkSize = 256;

struct CudaSpinGroupChunks {
  jams::MultiArray<int, 1> spin_indices;
  jams::MultiArray<int, 1> chunk_begin_offsets;
  jams::MultiArray<int, 1> chunk_end_offsets;
  jams::MultiArray<int, 1> chunk_group_indices;
  jams::MultiArray<int, 1> group_chunk_begin_offsets;
  jams::MultiArray<int, 1> group_chunk_end_offsets;
  int num_groups = 0;
  int num_chunks = 0;

  [[nodiscard]] std::size_t device_memory_bytes() const noexcept {
    return sizeof(int) * (
        spin_indices.size()
        + chunk_begin_offsets.size()
        + chunk_end_offsets.size()
        + chunk_group_indices.size()
        + group_chunk_begin_offsets.size()
        + group_chunk_end_offsets.size());
  }
};

[[nodiscard]]
CudaSpinGroupChunks make_cuda_spin_group_chunks(const std::vector<SpinGroup>& groups);

[[nodiscard]]
CudaSpinGroupChunks make_cuda_basis_spin_group_chunks(const Lattice& lattice, int num_spins);

CUDA_ONLY_IMPLEMENTATION(
void execute_cuda_grouped_spin_moment_reduction(
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

CUDA_ONLY_IMPLEMENTATION(
void execute_cuda_grouped_spin_sum_reduction(
    CudaStream& stream,
    int num_groups,
    int num_chunks,
    const int* chunk_begin_offsets,
    const int* chunk_end_offsets,
    const int* group_chunk_begin_offsets,
    const int* group_chunk_end_offsets,
    const int* spin_indices,
    const double* spins,
    double* chunk_sums,
    double* group_sum));

}  // namespace jams::monitors
#endif

#endif  // INCLUDED_JAMS_MONITORS_CUDA_GROUPED_SPIN_REDUCTION
// ----------------------------- END-OF-FILE ----------------------------------
