//
// Created by Joseph Barker on 2019-04-09.
//

#ifndef JAMS_TEST_SYNCED_MEMORY_H
#define JAMS_TEST_SYNCED_MEMORY_H

#include <array>
#include <complex>
#include <iterator>
#include <sstream>
#include <gmock/gmock-matchers.h>

#include "jams/containers/synced_memory.h"

#if HAS_CUDA
namespace {
bool have_synced_memory_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
}  // namespace
#endif

// fixture class
template <typename T>
class SynchedMemoryTest : public testing::Test {

};

typedef testing::Types<int, unsigned, double, std::complex<double>, std::array<int,3>, std::array<double,3>> SynchedMemoryTypes;
TYPED_TEST_SUITE(SynchedMemoryTest, SynchedMemoryTypes);

TYPED_TEST(SynchedMemoryTest, ctor) {
  using namespace jams;

  SyncedMemory<TypeParam> x;
  ASSERT_EQ(x.size(), 0);

  SyncedMemory<TypeParam> y(10);
  ASSERT_EQ(y.size(), 10);

  SyncedMemory<TypeParam> z(10, TypeParam{99});
  ASSERT_EQ(z.size(), 10);
}

TYPED_TEST(SynchedMemoryTest, size) {
  using namespace jams;
  using size_type = typename SyncedMemory<TypeParam>::size_type;

  SyncedMemory<TypeParam> x(10);
  ASSERT_EQ(x.size(), 10);
  ASSERT_EQ(x.bytes(), 10*sizeof(TypeParam));

  const auto expected_max_size =
      std::numeric_limits<size_type>::max() / sizeof(TypeParam);
  ASSERT_EQ(x.max_size(), expected_max_size);
}

TYPED_TEST(SynchedMemoryTest, accessors) {
  using namespace jams;

  SyncedMemory<TypeParam> x(10, TypeParam{99});
  ASSERT_NE(x.const_host_data(), nullptr);
  ASSERT_EQ(x.const_host_data(), x.mutable_host_data());

  ASSERT_EQ(x.const_host_data()[0], TypeParam{99});
  ASSERT_EQ(x.const_host_data()[9], TypeParam{99});
}

TYPED_TEST(SynchedMemoryTest, modifiers) {
  using namespace jams;

  SyncedMemory<TypeParam> x(10, TypeParam{99});

  x.zero();
  ASSERT_EQ(x.const_host_data()[0], TypeParam{0});
  ASSERT_EQ(x.const_host_data()[9], TypeParam{0});

  x.resize(20);
  ASSERT_EQ(x.size(), 20);

  x.clear();
  ASSERT_EQ(x.size(), 0);
  ASSERT_EQ(x.const_host_data(), nullptr);
}

TEST(SyncedMemoryRangeCtorTest, SupportsInputIterators) {
  std::istringstream input("1 2 3 4");
  std::istream_iterator<int> first(input);
  std::istream_iterator<int> last;

  jams::SyncedMemory<int> values(first, last);

  ASSERT_EQ(values.size(), 4u);
  const auto* host = values.const_host_data();
  ASSERT_NE(host, nullptr);
  EXPECT_EQ(host[0], 1);
  EXPECT_EQ(host[1], 2);
  EXPECT_EQ(host[2], 3);
  EXPECT_EQ(host[3], 4);
}

#if HAS_CUDA
TEST(SyncedMemoryGpuCopyCtorTest, PreservesDeviceMutatedState) {
  if (!have_synced_memory_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  jams::SyncedMemory<int> source(4, 7);
  ASSERT_NE(source.mutable_device_data(), nullptr);

  jams::SyncedMemory<int> copy(source);

  const auto* host = copy.const_host_data();
  ASSERT_NE(host, nullptr);
  EXPECT_EQ(copy.size(), 4u);
  for (std::size_t i = 0; i < copy.size(); ++i) {
    EXPECT_EQ(host[i], 7);
  }

  ASSERT_NE(copy.const_device_data(), nullptr);
}

TEST(SyncedMemoryGpuCopyAssignTest, ReplacesExistingDeviceBufferWhenSourceIsHostOnly) {
  if (!have_synced_memory_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  jams::SyncedMemory<int> destination(2, 1);
  ASSERT_NE(destination.mutable_device_data(), nullptr);

  jams::SyncedMemory<int> source(4, 9);

  destination = source;

  ASSERT_EQ(destination.size(), 4u);
  const auto* host = destination.const_host_data();
  ASSERT_NE(host, nullptr);
  for (std::size_t i = 0; i < destination.size(); ++i) {
    EXPECT_EQ(host[i], 9);
  }

  ASSERT_NE(destination.const_device_data(), nullptr);
}

TEST(SyncedMemoryGpuMoveAssignTest, TransfersOwnershipFromDeviceBackedSource) {
  if (!have_synced_memory_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  jams::SyncedMemory<int> destination(2, 1);
  ASSERT_NE(destination.mutable_device_data(), nullptr);

  jams::SyncedMemory<int> source(4, 5);
  ASSERT_NE(source.mutable_device_data(), nullptr);

  destination = std::move(source);

  ASSERT_EQ(destination.size(), 4u);
  const auto* host = destination.const_host_data();
  ASSERT_NE(host, nullptr);
  for (std::size_t i = 0; i < destination.size(); ++i) {
    EXPECT_EQ(host[i], 5);
  }

  EXPECT_EQ(source.size(), 0u);
  EXPECT_EQ(source.const_host_data(), nullptr);
}
#endif

#endif //JAMS_TEST_SYNCED_MEMORY_H
