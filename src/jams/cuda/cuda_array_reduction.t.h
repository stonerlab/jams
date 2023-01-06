#ifndef INCLUDED_JAMS_CUDA_ARRAY_REDUCTION_TEST_H
#define INCLUDED_JAMS_CUDA_ARRAY_REDUCTION_TEST_H

#include <jams/cuda/cuda_array_reduction.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <jams/helpers/array_ops.h>

///
/// @test
/// Tests that vector_field_reduce_cuda gives the same result as a reduction on
/// the CPU.
///
/// We use random data with a constant seed to avoid initializing with something
/// perversely simple like whole numbers.
///
/// @note
/// The CPU method MUST be of better accuracy than simply summing the numbers,
/// for example using a Kahan sum. Otherwise the round off errors will be
/// significant enough to fail a EXPECT_DOUBLE_EQ test. The cuda method is
/// less sensitive to round off errors because it is summing in small blocks.
///
TEST(CudaArrayReductionTest, vector_field_reduce_cuda) {
  using namespace testing;

  const int size = 1000000;

  std::random_device dev;
  std::seed_seq seed{12345, 67890, 5678, 6789, 6788};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist;

  jams::MultiArray<double, 2> x(size, 3);

  double counter = 0.0;
  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < 3; ++j) {
      x(i, j) = dist(rng);
      counter++;
    }
  }

  Vec3 cpu_result = jams::vector_field_reduce(x);
  Vec3 gpu_result = jams::vector_field_reduce_cuda(x);

  EXPECT_DOUBLE_EQ(cpu_result[0], gpu_result[0]);
  EXPECT_DOUBLE_EQ(cpu_result[1], gpu_result[1]);
  EXPECT_DOUBLE_EQ(cpu_result[2], gpu_result[2]);
}

///
/// @test
/// Tests that vector_field_indexed_reduce_cuda gives the same result as a
/// reduction on the CPU.
///
/// We use random data with a constant seed to avoid initializing with something
/// perversely simple like whole numbers.
///
/// @note
/// The CPU method MUST be of better accuracy than simply summing the numbers,
/// for example using a Kahan sum. Otherwise the round off errors will be
/// significant enough to fail a EXPECT_DOUBLE_EQ test. The cuda method is
/// less sensitive to round off errors because it is summing in small blocks.
///
TEST(CudaArrayReductionTest, vector_field_indexed_reduce_cuda) {
  using namespace testing;

  const int size = 100000;
  const int num_indices = 5000;

  std::random_device dev;
  std::seed_seq seed{12345, 67890, 5678, 6789, 6788};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist;
  std::uniform_int_distribution<int> int_dist(0,size);

  jams::MultiArray<double, 2> x(size, 3);

  double counter = 0.0;
  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < 3; ++j) {
      x(i, j) = dist(rng);
      counter++;
    }
  }

  jams::MultiArray<int, 1> indices(num_indices);
  for (auto i = 0; i < num_indices; ++i) {
    indices(i) = int_dist(rng);
  }

  Vec3 cpu_result = jams::vector_field_indexed_reduce(x, indices);
  Vec3 gpu_result = jams::vector_field_indexed_reduce_cuda(x, indices);

  EXPECT_DOUBLE_EQ(cpu_result[0], gpu_result[0]);
  EXPECT_DOUBLE_EQ(cpu_result[1], gpu_result[1]);
  EXPECT_DOUBLE_EQ(cpu_result[2], gpu_result[2]);
}

#endif