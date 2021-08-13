//
// Created by Joseph Barker on 2020-04-02.
//

#ifndef JAMS_TEST_NEARTREE_H
#define JAMS_TEST_NEARTREE_H

#include "jams/containers/neartree.h"

#include <random>

TEST(NearTreeTest, L1Norm) {
  using namespace jams;
  using namespace testing;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (auto repeats = 0; repeats < 10; ++repeats) {
    using Position = std::pair<Vec3, int>;

    const int num_positions = 1000;
    std::vector<Position> positions;

    for (auto i = 0; i < num_positions; ++i) {
      positions.push_back({{dist(rng), dist(rng), dist(rng)}, i});
    }

    using NeartreeFunctorType = std::function<double(const Position& a, const Position& b)>;

    auto l1_norm = [](const Position& a, const Position& b)->double {
      return sum(abs(a.first-b.first));
    };

    NearTree<Position, NeartreeFunctorType> near_tree(l1_norm, positions);

    const double epsilon = 1e-6;

    std::uniform_real_distribution<double> radius_dist(1.0, 10.0);

    for (auto i = 0; i < 100; ++i) {
      const double radius = radius_dist(rng);
      std::vector<Position> near_tree_neighbours = near_tree.find_in_radius(radius, positions[i], epsilon);

      std::vector<Position> brute_force_neighbours;
      for (auto j = 0; j < positions.size(); ++j) {
        // We need to use the same floating point comparison here as we use within the near_tree
        // otherwise we can very occasionally fail the test on the basis of the difference between
        // using '<' and doing a proper floating point comparison.
        if (!definately_greater_than(sum(abs( positions[i].first-positions[j].first)), radius, epsilon)) {
          brute_force_neighbours.push_back(positions[j]);
        }
      }

      ASSERT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
      ASSERT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin()));
    }
  }
}

TEST(NearTreeTest, L2Norm) {
  using namespace jams;
  using namespace testing;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (auto repeats = 0; repeats < 10; ++repeats) {
    using Position = std::pair<Vec3, int>;

    const int num_positions = 1000;
    std::vector<Position> positions;

    for (auto i = 0; i < num_positions; ++i) {
      positions.push_back({{dist(rng), dist(rng), dist(rng)}, i});
    }

    using NeartreeFunctorType = std::function<double(const Position& a, const Position& b)>;

    auto distance_metric = [](const Position& a, const Position& b)->double {
      return norm(a.first-b.first);
    };

    NearTree<Position, NeartreeFunctorType> near_tree(distance_metric, positions);

    const double epsilon = 1e-6;

    std::uniform_real_distribution<double> radius_dist(1.0, 10.0);

    for (auto i = 0; i < 100; ++i) {
      const double radius = radius_dist(rng);
      std::vector<Position> near_tree_neighbours = near_tree.find_in_radius(radius, positions[i], epsilon);

      std::vector<Position> brute_force_neighbours;
      for (auto j = 0; j < positions.size(); ++j) {
        // We need to use the same floating point comparison here as we use within the near_tree
        // otherwise we can very occasionally fail the test on the basis of the difference between
        // using '<' and doing a proper floating point comparison.
        if (!definately_greater_than(norm(positions[i].first - positions[j].first), radius, epsilon)) {
          brute_force_neighbours.push_back(positions[j]);
        }
      }

      ASSERT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
      ASSERT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin()));
    }
  }
}

#endif //JAMS_TEST_NEARTREE_H
