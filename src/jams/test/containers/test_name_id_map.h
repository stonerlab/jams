#ifndef JAMS_TEST_NAME_ID_MAP_H
#define JAMS_TEST_NAME_ID_MAP_H

#include <stdexcept>
#include <string>

#include "gtest/gtest.h"

#include "jams/containers/name_id_map.h"

TEST(NameIdMapTest, StoresDenseIdsAndBidirectionalNames) {
  NameIdMap<int> map;

  map.insert("alpha", 10);
  map.insert("beta", 20);

  EXPECT_EQ(map.size(), 2u);
  EXPECT_EQ(map[0], 10);
  EXPECT_EQ(map[1], 20);
  EXPECT_EQ(map["alpha"], 10);
  EXPECT_EQ(map["beta"], 20);
  EXPECT_EQ(map.name(0), "alpha");
  EXPECT_EQ(map.name(1), "beta");
  EXPECT_EQ(map.id("alpha"), 0u);
  EXPECT_EQ(map.id("beta"), 1u);
  EXPECT_TRUE(map.contains(0));
  EXPECT_TRUE(map.contains(1));
  EXPECT_FALSE(map.contains(2));
  EXPECT_TRUE(map.contains(std::string("alpha")));
  EXPECT_FALSE(map.contains(std::string("gamma")));
}

TEST(NameIdMapTest, RejectsDuplicateNamesWithoutChangingState) {
  NameIdMap<int> map;

  map.insert("alpha", 10);

  EXPECT_THROW(map.insert("alpha", 11), std::runtime_error);
  EXPECT_EQ(map.size(), 1u);
  EXPECT_EQ(map[0], 10);
  EXPECT_EQ(map.name(0), "alpha");
  EXPECT_EQ(map.id("alpha"), 0u);
}

TEST(NameIdMapTest, ClearResetsStoredMappings) {
  NameIdMap<int> map;

  map.insert("alpha", 10);
  map.insert("beta", 20);
  map.clear();

  EXPECT_EQ(map.size(), 0u);
  EXPECT_FALSE(map.contains(0));
  EXPECT_FALSE(map.contains(std::string("alpha")));

  map.insert("gamma", 30);
  EXPECT_EQ(map.size(), 1u);
  EXPECT_EQ(map.id("gamma"), 0u);
  EXPECT_EQ(map.name(0), "gamma");
  EXPECT_EQ(map[0], 30);
}

#endif // JAMS_TEST_NAME_ID_MAP_H
