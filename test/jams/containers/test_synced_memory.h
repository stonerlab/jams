//
// Created by Joseph Barker on 2019-04-09.
//

#ifndef JAMS_TEST_SYNCED_MEMORY_H
#define JAMS_TEST_SYNCED_MEMORY_H

#include "jams/containers/synced_memory.h"

TEST(SynchedMemoryTest, ctor) {
  using namespace jams;

  SyncedMemory<int> x;
  ASSERT_EQ(x.size(), 0);

  SyncedMemory<int> y(10);
  ASSERT_EQ(y.size(), 10);

  SyncedMemory<int> z(10, 99);
  ASSERT_EQ(z.size(), 10);
}

TEST(SynchedMemoryTest, accessors) {
  using namespace jams;

  SyncedMemory<int> x(10, 99);
  ASSERT_NE(x.const_host_data(), nullptr);
  ASSERT_EQ(x.const_host_data(), x.mutable_host_data());

  ASSERT_EQ(x.const_host_data()[0], 99);
  ASSERT_EQ(x.const_host_data()[9], 99);
}

TEST(SynchedMemoryTest, modifiers) {
  using namespace jams;

  SyncedMemory<int> x(10, 99);

  x.zero();
  ASSERT_EQ(x.const_host_data()[0], 0);

  x.resize(20);
  ASSERT_EQ(x.size(), 20);

  x.clear();
  ASSERT_EQ(x.size(), 0);
  ASSERT_EQ(x.const_host_data(), nullptr);
}

#endif //JAMS_TEST_SYNCED_MEMORY_H
