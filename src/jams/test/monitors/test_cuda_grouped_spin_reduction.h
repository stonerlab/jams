#ifndef JAMS_TEST_MONITORS_TEST_CUDA_GROUPED_SPIN_REDUCTION_H
#define JAMS_TEST_MONITORS_TEST_CUDA_GROUPED_SPIN_REDUCTION_H

#include "gtest/gtest.h"

#if HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <jams/containers/multiarray.h>
#include <jams/cuda/cuda_stream.h>
#include <jams/monitors/cuda_grouped_spin_reduction.h>

namespace jams::testing {
namespace {

inline bool cuda_grouped_spin_reduction_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

jams::monitors::SpinGroup make_test_spin_group(
    const std::string& name,
    const std::vector<int>& indices) {
  jams::monitors::SpinGroup group;
  group.name = name;
  group.indices.resize(indices.size());
  auto group_indices = group.indices.mutable_host_span();
  std::copy(indices.begin(), indices.end(), group_indices.begin());
  return group;
}

std::vector<int> spin_range(const int begin, const int end) {
  std::vector<int> indices;
  indices.reserve(static_cast<std::size_t>(end - begin));
  for (int spin = begin; spin < end; ++spin) {
    indices.push_back(spin);
  }
  return indices;
}

double spin_value(const int spin, const int component) {
  return 0.013 * static_cast<double>((spin + 3 * component) % 17)
      - 0.07 * static_cast<double>(component)
      + 0.001 * static_cast<double>(spin / 11);
}

jams::Real moment_value(const int spin) {
  return static_cast<jams::Real>(0.8 + 0.03 * static_cast<double>(spin % 9));
}

}  // namespace

TEST(CudaGroupedSpinReductionTest, WeightedAndUnweightedGroupedReductions) {
  if (!cuda_grouped_spin_reduction_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  constexpr int num_spins = 640;
  std::vector<jams::monitors::SpinGroup> groups;
  groups.reserve(4);
  groups.push_back(make_test_spin_group("empty", {}));
  groups.push_back(make_test_spin_group("partial", spin_range(0, 17)));
  groups.push_back(make_test_spin_group("multi_chunk", spin_range(17, 617)));
  groups.push_back(make_test_spin_group("tail", spin_range(617, num_spins)));

  auto chunks = jams::monitors::make_cuda_spin_group_chunks(groups);
  EXPECT_EQ(chunks.num_groups, 4);
  EXPECT_EQ(chunks.num_chunks, 5);

  jams::MultiArray<double, 2> spins(num_spins, 3);
  jams::MultiArray<jams::Real, 1> moments(num_spins);
  auto spin_view = spins.mutable_host_view();
  auto moment_span = moments.mutable_host_span();
  for (int spin = 0; spin < num_spins; ++spin) {
    moment_span[spin] = moment_value(spin);
    for (int component = 0; component < 3; ++component) {
      spin_view(spin, component) = spin_value(spin, component);
    }
  }

  std::vector<std::array<double, 3>> expected_unweighted(groups.size(), {0.0, 0.0, 0.0});
  std::vector<std::array<double, 3>> expected_weighted(groups.size(), {0.0, 0.0, 0.0});
  for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
    for (const int spin : groups[group_index].indices_span()) {
      for (int component = 0; component < 3; ++component) {
        const double value = spin_value(spin, component);
        expected_unweighted[group_index][component] += value;
        expected_weighted[group_index][component] +=
            static_cast<double>(moment_value(spin)) * value;
      }
    }
  }

  CudaStream stream;
  jams::MultiArray<double, 2> weighted(groups.size(), 3);
  jams::monitors::execute_cuda_grouped_spin_moment_reduction(
      stream,
      chunks.num_groups,
      chunks.num_chunks,
      chunks.chunk_begin_offsets.device_data(),
      chunks.chunk_end_offsets.device_data(),
      chunks.chunk_group_indices.device_data(),
      chunks.spin_indices.device_data(),
      spins.device_data(),
      moments.device_data(),
      weighted.mutable_device_data());

  jams::MultiArray<double, 2> unweighted(groups.size(), 3);
  jams::MultiArray<double, 2> chunk_sums(chunks.num_chunks, 3);
  jams::monitors::execute_cuda_grouped_spin_sum_reduction(
      stream,
      chunks.num_groups,
      chunks.num_chunks,
      chunks.chunk_begin_offsets.device_data(),
      chunks.chunk_end_offsets.device_data(),
      chunks.group_chunk_begin_offsets.device_data(),
      chunks.group_chunk_end_offsets.device_data(),
      chunks.spin_indices.device_data(),
      spins.device_data(),
      chunk_sums.mutable_device_data(),
      unweighted.mutable_device_data());
  stream.synchronize();

  const auto weighted_host = weighted.host_view();
  const auto unweighted_host = unweighted.host_view();
  for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
    for (int component = 0; component < 3; ++component) {
      EXPECT_NEAR(
          weighted_host(group_index, component),
          expected_weighted[group_index][component],
          1.0e-9);
      EXPECT_NEAR(
          unweighted_host(group_index, component),
          expected_unweighted[group_index][component],
          1.0e-9);
    }
  }
}

}  // namespace jams::testing

#endif  // HAS_CUDA

#endif  // JAMS_TEST_MONITORS_TEST_CUDA_GROUPED_SPIN_REDUCTION_H
