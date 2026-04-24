//
// Created by Joseph Barker on 2019-04-09.
//

#ifndef JAMS_TEST_SYNCED_MEMORY_H
#define JAMS_TEST_SYNCED_MEMORY_H

#include <array>
#include <complex>
#include <iterator>
#include <gmock/gmock-matchers.h>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "jams/containers/synced_memory.h"

#if HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {
bool synced_memory_cuda_device_available() {
#if HAS_CUDA
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}
}

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

  SyncedMemory<TypeParam> x(10);
  ASSERT_EQ(x.size(), 10);
  ASSERT_EQ(x.bytes(), 10*sizeof(TypeParam));

  // if we can't address the memory space with int then we're
  // in trouble
  ASSERT_GE(x.max_size(), std::numeric_limits<int>::max());
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

TEST(SyncedMemoryApiTest, TraitsAndNoexceptContract) {
  using Memory = jams::SyncedMemory<int>;

  static_assert(std::is_default_constructible<Memory>::value, "");
  static_assert(std::is_copy_constructible<Memory>::value, "");
  static_assert(std::is_copy_assignable<Memory>::value, "");
  static_assert(std::is_nothrow_destructible<Memory>::value, "");
  static_assert(std::is_nothrow_move_constructible<Memory>::value, "");
  static_assert(std::is_nothrow_move_assignable<Memory>::value, "");
  static_assert(noexcept(swap(std::declval<Memory&>(), std::declval<Memory&>())), "");
  static_assert(noexcept(std::declval<const Memory&>().empty()), "");
  static_assert(noexcept(std::declval<const Memory&>().size()), "");
  static_assert(noexcept(std::declval<const Memory&>().bytes()), "");
  static_assert(noexcept(std::declval<const Memory&>().host_valid()), "");
  static_assert(noexcept(std::declval<const Memory&>().device_valid()), "");
  static_assert(!noexcept(std::declval<const Memory&>().max_size()), "");
  static_assert(!noexcept(std::declval<const Memory&>().host_data()), "");
  static_assert(!noexcept(std::declval<const Memory&>().device_data()), "");
  static_assert(!noexcept(std::declval<Memory&>().mutable_host_data()), "");
  static_assert(!noexcept(std::declval<Memory&>().mutable_device_data()), "");
}

TEST(SyncedMemoryApiTest, DefaultAndSizedConstructionAreLazy) {
  using namespace jams;

  SyncedMemory<int> empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0u);
  EXPECT_EQ(empty.bytes(), 0u);
  EXPECT_FALSE(empty.host_valid());
  EXPECT_FALSE(empty.device_valid());
  EXPECT_EQ(empty.host_data(), nullptr);
  EXPECT_EQ(empty.device_data(), nullptr);

  SyncedMemory<int> sized(4);
  EXPECT_FALSE(sized.empty());
  EXPECT_EQ(sized.size(), 4u);
  EXPECT_EQ(sized.bytes(), 4u * sizeof(int));
  EXPECT_FALSE(sized.host_valid());
  EXPECT_FALSE(sized.device_valid());
}

TEST(SyncedMemoryApiTest, FillConstructionCreatesCurrentHostDataOnly) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> memory(4, 7);
  EXPECT_TRUE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());

  const SyncedMemory<int>& const_memory = memory;
  const int* host = const_memory.host_data();
  ASSERT_NE(host, nullptr);
  EXPECT_THAT(std::vector<int>(host, host + memory.size()), ElementsAre(7, 7, 7, 7));
  EXPECT_TRUE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());
}

TEST(SyncedMemoryApiTest, MutableHostAccessInvalidatesOnlyDeviceCurrentness) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> memory(3, 1);

#if HAS_CUDA
  if (synced_memory_cuda_device_available()) {
    ASSERT_NE(memory.device_data(), nullptr);
    EXPECT_TRUE(memory.host_valid());
    EXPECT_TRUE(memory.device_valid());
  }
#endif

  int* host = memory.mutable_host_data();
  ASSERT_NE(host, nullptr);
  host[0] = 4;
  host[1] = 5;
  host[2] = 6;

  EXPECT_TRUE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());

  const SyncedMemory<int>& const_memory = memory;
  const int* readback = const_memory.host_data();
  EXPECT_EQ(readback, host);
  EXPECT_THAT(std::vector<int>(readback, readback + memory.size()), ElementsAre(4, 5, 6));
  EXPECT_TRUE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());
}

#if HAS_CUDA
TEST(SyncedMemoryApiTest, ConstDeviceReadSynchronizesHostChangesWithoutInvalidatingHost) {
  using namespace jams;

  if (!synced_memory_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  SyncedMemory<int> memory(3, 0);
  int* host = memory.mutable_host_data();
  host[0] = 11;
  host[1] = 12;
  host[2] = 13;
  EXPECT_TRUE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());

  const SyncedMemory<int>& const_memory = memory;
  const int* device = const_memory.device_data();
  ASSERT_NE(device, nullptr);
  EXPECT_TRUE(memory.host_valid());
  EXPECT_TRUE(memory.device_valid());

  int copied[3] = {};
  ASSERT_EQ(cudaMemcpy(copied, device, sizeof(copied), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_THAT(copied, testing::ElementsAre(11, 12, 13));
  EXPECT_TRUE(memory.host_valid());
  EXPECT_TRUE(memory.device_valid());
}

TEST(SyncedMemoryApiTest, MutableDeviceAccessInvalidatesHostUntilConstHostRead) {
  using namespace jams;
  using namespace testing;

  if (!synced_memory_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  SyncedMemory<int> memory(3, 1);
  int device_values[3] = {21, 22, 23};
  int* device = memory.mutable_device_data();
  ASSERT_NE(device, nullptr);
  ASSERT_EQ(cudaMemcpy(device, device_values, sizeof(device_values), cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_FALSE(memory.host_valid());
  EXPECT_TRUE(memory.device_valid());

  const SyncedMemory<int>& const_memory = memory;
  const int* host = const_memory.host_data();
  ASSERT_NE(host, nullptr);
  EXPECT_THAT(std::vector<int>(host, host + memory.size()), ElementsAre(21, 22, 23));
  EXPECT_TRUE(memory.host_valid());
  EXPECT_TRUE(memory.device_valid());
}
#endif

TEST(SyncedMemoryApiTest, InputIteratorConstructionConsumesSinglePassRangeOnce) {
  using namespace jams;
  using namespace testing;

  std::istringstream input("3 1 4 1 5");
  std::istream_iterator<int> first(input);
  std::istream_iterator<int> last;

  SyncedMemory<int> memory(first, last);
  EXPECT_EQ(memory.size(), 5u);
  EXPECT_TRUE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());

  const int* host = memory.host_data();
  ASSERT_NE(host, nullptr);
  EXPECT_THAT(std::vector<int>(host, host + memory.size()), ElementsAre(3, 1, 4, 1, 5));
}

TEST(SyncedMemoryApiTest, CopyConstructionCopiesCurrentLogicalValue) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> source(3, 1);

#if HAS_CUDA
  if (synced_memory_cuda_device_available()) {
    ASSERT_NE(source.device_data(), nullptr);
  }
#endif

  int* source_host = source.mutable_host_data();
  source_host[0] = 8;
  source_host[1] = 9;
  source_host[2] = 10;

  const SyncedMemory<int> copy(source);
  EXPECT_EQ(copy.size(), source.size());
  EXPECT_TRUE(copy.host_valid());
  EXPECT_FALSE(copy.device_valid());

  const int* copied_host = copy.host_data();
  ASSERT_NE(copied_host, nullptr);
  EXPECT_THAT(std::vector<int>(copied_host, copied_host + copy.size()), ElementsAre(8, 9, 10));

  source_host[0] = -1;
  EXPECT_EQ(copied_host[0], 8);
}

TEST(SyncedMemoryApiTest, CopyAssignmentReplacesSizeStateAndLogicalValue) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> source(5, 2);
  int* source_host = source.mutable_host_data();
  for (std::size_t i = 0; i < source.size(); ++i) {
    source_host[i] = static_cast<int>(10 + i);
  }

  SyncedMemory<int> target(2, -1);
#if HAS_CUDA
  if (synced_memory_cuda_device_available()) {
    ASSERT_NE(target.device_data(), nullptr);
  }
#endif

  target = source;
  EXPECT_EQ(target.size(), 5u);
  EXPECT_TRUE(target.host_valid());
  EXPECT_FALSE(target.device_valid());

  const int* target_host = target.host_data();
  ASSERT_NE(target_host, nullptr);
  EXPECT_THAT(std::vector<int>(target_host, target_host + target.size()),
              ElementsAre(10, 11, 12, 13, 14));

  source_host[0] = 99;
  EXPECT_EQ(target_host[0], 10);
}

TEST(SyncedMemoryApiTest, CopyAssignmentReusesHostStorageWhenSizeMatches) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> source(3, 1);
  source.mutable_host_data()[0] = 4;
  source.mutable_host_data()[1] = 5;
  source.mutable_host_data()[2] = 6;

  SyncedMemory<int> target(3, -1);
  int* original_target_host = target.mutable_host_data();
  ASSERT_NE(original_target_host, nullptr);

  target = source;
  EXPECT_EQ(target.mutable_host_data(), original_target_host);
  EXPECT_TRUE(target.host_valid());
  EXPECT_FALSE(target.device_valid());
  EXPECT_THAT(std::vector<int>(target.host_data(), target.host_data() + target.size()),
              ElementsAre(4, 5, 6));
}

#if HAS_CUDA
TEST(SyncedMemoryApiTest, CopyConstructionCopiesDeviceOnlySourceThroughDevice) {
  using namespace jams;
  using namespace testing;

  if (!synced_memory_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  SyncedMemory<int> source(3, 0);
  int values[3] = {31, 32, 33};
  int* source_device = source.mutable_device_data();
  ASSERT_NE(source_device, nullptr);
  ASSERT_EQ(cudaMemcpy(source_device, values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_FALSE(source.host_valid());
  EXPECT_TRUE(source.device_valid());

  SyncedMemory<int> copy(source);
  EXPECT_FALSE(copy.host_valid());
  EXPECT_TRUE(copy.device_valid());

  const int* host = copy.host_data();
  ASSERT_NE(host, nullptr);
  EXPECT_THAT(std::vector<int>(host, host + copy.size()), ElementsAre(31, 32, 33));
}

TEST(SyncedMemoryApiTest, CopyAssignmentReusesDeviceStorageWhenSizeMatches) {
  using namespace jams;
  using namespace testing;

  if (!synced_memory_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  SyncedMemory<int> source(3, 0);
  int values[3] = {41, 42, 43};
  int* source_device = source.mutable_device_data();
  ASSERT_NE(source_device, nullptr);
  ASSERT_EQ(cudaMemcpy(source_device, values, sizeof(values), cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_FALSE(source.host_valid());
  EXPECT_TRUE(source.device_valid());

  SyncedMemory<int> target(3, 0);
  int* original_target_device = target.mutable_device_data();
  ASSERT_NE(original_target_device, nullptr);

  target = source;
  EXPECT_EQ(target.mutable_device_data(), original_target_device);
  EXPECT_FALSE(target.host_valid());
  EXPECT_TRUE(target.device_valid());

  const int* host = target.host_data();
  ASSERT_NE(host, nullptr);
  EXPECT_THAT(std::vector<int>(host, host + target.size()), ElementsAre(41, 42, 43));
}
#endif

TEST(SyncedMemoryApiTest, MoveConstructionTransfersOwnershipAndEmptiesSource) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> source(3, 6);
  const int* original_host = source.host_data();
  ASSERT_NE(original_host, nullptr);

  SyncedMemory<int> moved(std::move(source));
  EXPECT_TRUE(source.empty());
  EXPECT_EQ(source.host_data(), nullptr);
  EXPECT_EQ(moved.size(), 3u);
  EXPECT_TRUE(moved.host_valid());

  const int* moved_host = moved.host_data();
  EXPECT_EQ(moved_host, original_host);
  EXPECT_THAT(std::vector<int>(moved_host, moved_host + moved.size()), ElementsAre(6, 6, 6));
}

TEST(SyncedMemoryApiTest, MoveAssignmentReleasesOldStorageTransfersOwnershipAndEmptiesSource) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> source(4, 12);
  const int* original_host = source.host_data();
  ASSERT_NE(original_host, nullptr);

  SyncedMemory<int> target(2, -7);
  target = std::move(source);

  EXPECT_TRUE(source.empty());
  EXPECT_EQ(source.host_data(), nullptr);
  EXPECT_EQ(target.size(), 4u);
  EXPECT_TRUE(target.host_valid());

  const int* target_host = target.host_data();
  EXPECT_EQ(target_host, original_host);
  EXPECT_THAT(std::vector<int>(target_host, target_host + target.size()), ElementsAre(12, 12, 12, 12));
}

TEST(SyncedMemoryApiTest, SwapExchangesStorageAndValidity) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> lhs(2, 1);
  SyncedMemory<int> rhs(3, 4);
  lhs.mutable_host_data()[1] = 2;
  rhs.mutable_host_data()[1] = 5;
  rhs.mutable_host_data()[2] = 6;

  swap(lhs, rhs);

  EXPECT_EQ(lhs.size(), 3u);
  EXPECT_EQ(rhs.size(), 2u);
  EXPECT_TRUE(lhs.host_valid());
  EXPECT_TRUE(rhs.host_valid());
  EXPECT_THAT(std::vector<int>(lhs.host_data(), lhs.host_data() + lhs.size()), ElementsAre(4, 5, 6));
  EXPECT_THAT(std::vector<int>(rhs.host_data(), rhs.host_data() + rhs.size()), ElementsAre(1, 2));
}

TEST(SyncedMemoryApiTest, ResizeAndClearDiscardCurrentness) {
  using namespace jams;

  SyncedMemory<int> memory(3, 9);
  ASSERT_TRUE(memory.host_valid());

  memory.resize(5);
  EXPECT_EQ(memory.size(), 5u);
  EXPECT_FALSE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());

  memory.clear();
  EXPECT_TRUE(memory.empty());
  EXPECT_EQ(memory.size(), 0u);
  EXPECT_FALSE(memory.host_valid());
  EXPECT_FALSE(memory.device_valid());
  EXPECT_EQ(memory.host_data(), nullptr);
  EXPECT_EQ(memory.device_data(), nullptr);
}

TEST(SyncedMemoryApiTest, ZeroKeepsOnlyAllocatedCurrentMemorySpacesCurrent) {
  using namespace jams;
  using namespace testing;

  SyncedMemory<int> host_only(3, 5);
  host_only.zero();
  EXPECT_TRUE(host_only.host_valid());
  EXPECT_FALSE(host_only.device_valid());
  EXPECT_THAT(std::vector<int>(host_only.host_data(), host_only.host_data() + host_only.size()),
              ElementsAre(0, 0, 0));

#if HAS_CUDA
  if (synced_memory_cuda_device_available()) {
    SyncedMemory<int> host_and_device(3, 5);
    ASSERT_NE(host_and_device.device_data(), nullptr);
    host_and_device.zero();
    EXPECT_TRUE(host_and_device.host_valid());
    EXPECT_TRUE(host_and_device.device_valid());
    EXPECT_THAT(std::vector<int>(host_and_device.host_data(),
                                 host_and_device.host_data() + host_and_device.size()),
                ElementsAre(0, 0, 0));
  }
#endif
}

#endif //JAMS_TEST_SYNCED_MEMORY_H
