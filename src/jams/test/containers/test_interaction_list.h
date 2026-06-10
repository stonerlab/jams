#ifndef JAMS_TEST_INTERACTION_LIST_H
#define JAMS_TEST_INTERACTION_LIST_H

#include <cstddef>
#include <type_traits>

#include <gtest/gtest.h>

#include "jams/containers/interaction_list.h"

TEST(InteractionListTest, UsesSizeTForListPositions) {
  static_assert(std::is_same_v<jams::InteractionList<double, 2>::size_type, std::size_t>);
}

TEST(InteractionListTest, CountsInteractionsByFirstIndex) {
  jams::InteractionList<double, 2> list;

  list.insert({0, 1}, 1.0);
  list.insert({0, 2}, 2.0);
  list.insert({3, 1}, 3.0);

  EXPECT_EQ(list.size(), 3u);
  EXPECT_EQ(list.num_interactions(), 3u);
  EXPECT_EQ(list.num_interactions(0), 2u);
  EXPECT_EQ(list.num_interactions(1), 0u);
  EXPECT_EQ(list.num_interactions(3), 1u);
}

#endif // JAMS_TEST_INTERACTION_LIST_H
