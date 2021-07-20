//
// Created by Joseph Barker on 2020-04-02.
//

#ifndef JAMS_TEST_NEARTREE_H
#define JAMS_TEST_NEARTREE_H

#include "jams/containers/neartree.h"

TEST(NearTreeTest, L1Norm) {
  using namespace jams;
  using namespace testing;

  using Position = std::pair<Vec3, int>;
  std::vector<Position> positions;

  for (auto i = 0; i < 1000; ++i) {
    positions.push_back({
                            {rand()/double(RAND_MAX), rand()/double(RAND_MAX), rand()/double(RAND_MAX)}, i});
  }

  using NeartreeFunctorType = std::function<double(const Position& a, const Position& b)>;

  auto l1_norm = [](const Position& a, const Position& b)->double {
      return sum(abs(a.first-b.first));
  };

  NearTree<Position, NeartreeFunctorType> near_tree(l1_norm, positions);


  const double epsilon = 1e-6;
  for (auto i = 0; i < 100; ++i) {
    const double radius = rand()/double(RAND_MAX);
    std::vector<Position> near_tree_neighbours = near_tree.find_in_radius(radius, positions[i], epsilon);

    std::vector<Position> brute_force_neighbours;
    for (auto j = 0; j < positions.size(); ++j) {
      if (sum(abs( positions[i].first-positions[j].first)) < radius) {
        brute_force_neighbours.push_back(positions[j]);
      }
    }

    ASSERT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
    ASSERT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin()));
  }
}

TEST(NearTreeTest, L2Norm) {
  using namespace jams;
  using namespace testing;

  using Position = std::pair<Vec3, int>;
  std::vector<Position> positions;

  for (auto i = 0; i < 1000; ++i) {
    positions.push_back({
                            {rand()/double(RAND_MAX), rand()/double(RAND_MAX), rand()/double(RAND_MAX)}, i});
  }

  using NeartreeFunctorType = std::function<double(const Position& a, const Position& b)>;

  auto distance_metric = [](const Position& a, const Position& b)->double {
      return norm(a.first-b.first);
  };

  NearTree<Position, NeartreeFunctorType> near_tree(distance_metric, positions);

  const double epsilon = 1e-6;

  for (auto i = 0; i < 100; ++i) {
    const double radius = rand()/double(RAND_MAX);
    std::vector<Position> near_tree_neighbours = near_tree.find_in_radius(radius, positions[i], epsilon);

    std::vector<Position> brute_force_neighbours;
    for (auto j = 0; j < positions.size(); ++j) {
      if (norm(positions[i].first - positions[j].first) < radius) {
        brute_force_neighbours.push_back(positions[j]);
      }
    }

    ASSERT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
    ASSERT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin()));
  }
}

#endif //JAMS_TEST_NEARTREE_H
