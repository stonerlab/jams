//
// Created by Joseph Barker on 2019-04-09.
//

#ifndef JAMS_TEST_SYNCED_MEMORY_H
#define JAMS_TEST_SYNCED_MEMORY_H

#include <array>
#include <complex>
#include <gmock/gmock-matchers.h>

#include "jams/containers/synced_memory.h"

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
  ASSERT_EQ(x.memory(), 10*sizeof(TypeParam));

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

#endif //JAMS_TEST_SYNCED_MEMORY_H
