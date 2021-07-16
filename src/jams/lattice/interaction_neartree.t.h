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

  Vec3 a = {1.0, 0.0, 0.0};
  Vec3 b = {0.0, 1.0, 0.0};
  Vec3 c = {0.0, 0.0, 1.0};
  Vec3b pbc = {false, false, false};
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  jams::InteractionNearTree near_tree(10*a, 10*b, 10*c, pbc, r_cutoff, epsilon);

  std::vector<Vec3> sites;
  for (auto i = 0; i < 10; ++i) {
    for (auto j = 0; j < 10; ++j) {
      for (auto k = 0; k < 10; ++k) {
        sites.push_back(i*a + j*b + k*c);
      }
    }
  }

  near_tree.insert_sites(sites);


  for (const auto& r_i : sites) {

    int num_nbrs_bruteforce = 0;
    for (const auto& r_j : sites) {
      auto r_ij = jams::minimum_image(10*a, 10*b, 10*c, pbc, r_i, r_j, epsilon);

      if (norm(r_ij) < epsilon) continue;

      if (definately_greater_than(norm(r_ij), r_cutoff, epsilon)) continue;

      num_nbrs_bruteforce++;
    }
    EXPECT_EQ(near_tree.num_neighbours(r_i, r_cutoff), num_nbrs_bruteforce);

  }

}

TEST(InteractionNeartreeTest, neighbours_pbc_simple) {
  using namespace testing;

  Vec3 a = {1.0, 0.0, 0.0};
  Vec3 b = {0.0, 1.0, 0.0};
  Vec3 c = {0.0, 0.0, 1.0};
  Vec3b pbc = {true, true, true};
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  jams::InteractionNearTree near_tree(10*a, 10*b, 10*c, pbc, r_cutoff, epsilon);

  std::vector<Vec3> sites;
  for (auto i = 0; i < 10; ++i) {
    for (auto j = 0; j < 10; ++j) {
      for (auto k = 0; k < 10; ++k) {
        sites.push_back(i*a + j*b + k*c);
      }
    }
  }

  near_tree.insert_sites(sites);


  for (const auto& r_i : sites) {

    int num_nbrs_bruteforce = 0;
    for (const auto& r_j : sites) {
      auto r_ij = jams::minimum_image(10*a, 10*b, 10*c, pbc, r_i, r_j, epsilon);

      if (norm(r_ij) < epsilon) continue;

      if (definately_greater_than(norm(r_ij), r_cutoff, epsilon)) continue;

      num_nbrs_bruteforce++;
    }
    EXPECT_EQ(near_tree.num_neighbours(r_i, r_cutoff), num_nbrs_bruteforce);

  }

}

TEST(InteractionNeartreeTest, neighbours_no_pbc) {
  using namespace testing;

  Vec3 a = {0.5, 0.0, 0.0};
  Vec3 b = {-0.25, 0.4330127019, 0.0};
  Vec3 c = {0.0, 0.2886751346, 0.4082482905};
  Vec3b pbc = {false, false, false};
  double r_cutoff = 3.0;
  double epsilon = 1e-5;

  jams::InteractionNearTree near_tree(10*a, 10*b, 10*c, pbc, r_cutoff, epsilon);

  std::vector<Vec3> sites;
  for (auto i = 0; i < 10; ++i) {
    for (auto j = 0; j < 10; ++j) {
      for (auto k = 0; k < 10; ++k) {
        sites.push_back(i*a + j*b + k*c);
      }
    }
  }

  near_tree.insert_sites(sites);


  for (const auto& r_i : sites) {

    int num_nbrs_bruteforce = 0;
    for (const auto& r_j : sites) {
      auto r_ij = jams::minimum_image(10*a, 10*b, 10*c, pbc, r_i, r_j, epsilon);

      if (norm(r_ij) < 1e-3) continue;

      if (definately_greater_than(norm(r_ij), r_cutoff, epsilon)) continue;

      num_nbrs_bruteforce++;
    }


    EXPECT_EQ(near_tree.num_neighbours(r_i, r_cutoff), num_nbrs_bruteforce);

  }

}

#endif //JAMS_LATTICE_INTERACTION_NEARTREE_TEST_H
