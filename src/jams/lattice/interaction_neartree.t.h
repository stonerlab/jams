#ifndef JAMS_LATTICE_INTERACTION_NEARTREE_TEST_H
#define JAMS_LATTICE_INTERACTION_NEARTREE_TEST_H

#include <jams/lattice/interaction_neartree.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

///
/// @test
/// Tests that the minimum_image_smith_method function gives the shortest r_ij
/// for results where |r_ij| is less than the inradius of the cell.
///

TEST(InteractionNeartreeTest, neighbours_no_pbc_simple) {
  using namespace testing;
  using Position = std::pair<Vec3, int>;

  const int N = 5;

  Vec3 a = {1.0, 0.0, 0.0};
  Vec3 b = {0.0, 1.0, 0.0};
  Vec3 c = {0.0, 0.0, 1.0};
  Vec3b pbc = {false, false, false};
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  jams::InteractionNearTree near_tree(N*a, N*b, N*c, pbc, r_cutoff, epsilon);

  std::vector<Position> positions;
  std::vector<Vec3> sites;
  int count = 0;
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < N; ++j) {
      for (auto k = 0; k < N; ++k) {
        positions.emplace_back(i * a + j * b + k * c, count);
        sites.push_back(i * a + j * b + k * c);
        count++;
      }
    }
  }

  near_tree.insert_sites(sites);

  for (const auto &r_i : positions) {

    std::vector<Position> near_tree_neighbours = near_tree.neighbours(r_i.first, r_cutoff);

    std::vector<Position> brute_force_neighbours;

    for (const auto &r_j : positions) {
      const auto r_ij = r_i.first - r_j.first;
      if (norm(r_ij) < epsilon) continue;
      if (definately_greater_than(norm(r_ij), r_cutoff, epsilon)) continue;
      brute_force_neighbours.push_back(r_j);
    }

    auto positions_are_equivalent = [epsilon](const Position& a, const Position& b){
      return approximately_equal(a.first, b.first, epsilon) && a.second == b.second;
    };


    EXPECT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
    EXPECT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin(), positions_are_equivalent));
  }
}


TEST(InteractionNeartreeTest, neighbours_pbc_simple) {
  using namespace testing;
  using Position = std::pair<Vec3, int>;

  const int N = 5;

  Vec3 a = {1.0, 0.0, 0.0};
  Vec3 b = {0.0, 1.0, 0.0};
  Vec3 c = {0.0, 0.0, 1.0};
  Vec3b pbc = {true, true, true};
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  jams::InteractionNearTree near_tree(N*a, N*b, N*c, pbc, r_cutoff, epsilon);

  std::vector<Position> positions;
  std::vector<Vec3> sites;
  int count = 0;
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < N; ++j) {
      for (auto k = 0; k < N; ++k) {
        positions.emplace_back(i * a + j * b + k * c, count);
        sites.push_back(i * a + j * b + k * c);
        count++;
      }
    }
  }

  near_tree.insert_sites(sites);

  for (const auto &r_i : positions) {

    std::vector<Position> near_tree_neighbours = near_tree.neighbours(r_i.first, r_cutoff);

    for (auto &r_j : near_tree_neighbours) {
      r_j.first = r_i.first - r_j.first;
    }

    std::vector<Position> brute_force_neighbours;

    for (const auto &r_j : positions) {
      auto r_ij = jams::minimum_image(N*a, N*b, N*c, pbc, r_i.first,
                                      r_j.first, epsilon);
      if (norm(r_ij) < epsilon) continue;
      if (definately_greater_than(norm(r_ij), r_cutoff, epsilon)) continue;
      brute_force_neighbours.emplace_back(r_ij, r_j.second);
    }

    auto positions_are_equivalent = [epsilon](const Position& a, const Position& b){
        return approximately_equal(a.first, b.first, epsilon) && a.second == b.second;
    };

    EXPECT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
    EXPECT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin(), positions_are_equivalent));
  }
}


TEST(InteractionNeartreeTest, neighbours_no_pbc_complicated) {
  using namespace testing;
  using Position = std::pair<Vec3, int>;

  const int N = 5;

  Vec3 a = {0.5, 0.0, 0.0};
  Vec3 b = {-0.25, 0.4330127019, 0.0};
  Vec3 c = {0.0, 0.2886751346, 0.4082482905};
  Vec3b pbc = {false, false, false};
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  jams::InteractionNearTree near_tree(N*a, N*b, N*c, pbc, r_cutoff, epsilon);

  std::vector<Position> positions;
  std::vector<Vec3> sites;
  int count = 0;
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < N; ++j) {
      for (auto k = 0; k < N; ++k) {
        positions.emplace_back(i * a + j * b + k * c, count);
        sites.push_back(i * a + j * b + k * c);
        count++;
      }
    }
  }

  near_tree.insert_sites(sites);

  for (const auto &r_i : positions) {

    std::vector<Position> near_tree_neighbours = near_tree.neighbours(r_i.first, r_cutoff);

    std::vector<Position> brute_force_neighbours;

    for (const auto &r_j : positions) {
      const auto r_ij = r_i.first - r_j.first;
      if (norm(r_ij) < epsilon) continue;
      if (definately_greater_than(norm(r_ij), r_cutoff, epsilon)) continue;
      brute_force_neighbours.push_back(r_j);
    }

    auto positions_are_equivalent = [epsilon](const Position& a, const Position& b){
        return approximately_equal(a.first, b.first, epsilon) && a.second == b.second;
    };


    EXPECT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
    EXPECT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin(), positions_are_equivalent));
  }
}


TEST(InteractionNeartreeTest, neighbours_pbc_complicated) {
  using namespace testing;
  using Position = std::pair<Vec3, int>;

  const int N = 12;

  Vec3 a = {0.5, 0.0, 0.0};
  Vec3 b = {-0.25, 0.4330127019, 0.0};
  Vec3 c = {0.0, 0.2886751346, 0.4082482905};
  Vec3b pbc = {true, true, true};
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  ASSERT_LE(r_cutoff, jams::maths::parallelepiped_inradius(N*a, N*b, N*c));

  jams::InteractionNearTree near_tree(N*a, N*b, N*c, pbc, r_cutoff, epsilon);

  std::vector<Position> positions;
  std::vector<Vec3> sites;
  int count = 0;
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < N; ++j) {
      for (auto k = 0; k < N; ++k) {
        positions.emplace_back(i * a + j * b + k * c, count);
        sites.push_back(i * a + j * b + k * c);
        count++;
      }
    }
  }

  near_tree.insert_sites(sites);

  for (const auto &r_i : positions) {

    std::vector<Position> near_tree_neighbours = near_tree.neighbours(r_i.first, r_cutoff);

    for (auto &r_j : near_tree_neighbours) {
      r_j.first = r_i.first - r_j.first;
    }

    std::vector<Position> brute_force_neighbours;

    for (const auto &r_j : positions) {
      auto r_ij = jams::minimum_image(N*a, N*b, N*c, pbc, r_i.first,
                                      r_j.first, epsilon);
      if (norm(r_ij) < epsilon) continue;
      if (definately_greater_than(norm(r_ij), r_cutoff, epsilon)) continue;
      brute_force_neighbours.emplace_back(r_ij, r_j.second);
    }

    auto positions_are_equivalent = [epsilon](const Position& a, const Position& b){
        return approximately_equal(a.first, b.first, epsilon) && a.second == b.second;
    };

    EXPECT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
    EXPECT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin(), positions_are_equivalent));
  }
}


#endif //JAMS_LATTICE_INTERACTION_NEARTREE_TEST_H
