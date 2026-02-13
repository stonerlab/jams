#ifndef JAMS_LATTICE_INTERACTION_NEARTREE_TEST_H
#define JAMS_LATTICE_INTERACTION_NEARTREE_TEST_H

#include <jams/lattice/interaction_neartree.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <memory>
#include <stdexcept>

class InteractionNeartreeTest : public ::testing::Test {
protected:
    using Position = std::pair<Vec3, int>;

    InteractionNeartreeTest() {};

    ~InteractionNeartreeTest() = default;

    void SetUp(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3b& pbc, const int& supercell_size, const double& r_cutoff, const double& epsilon) {

      a_ = a; b_ = b; c_ = c;
      pbc_ = pbc;
      supercell_size_ = supercell_size;
      r_cutoff_ = r_cutoff;
      epsilon_ = epsilon;

      // To do the brute force calculation we use minimum_image which can only
      // find neighbours within the supercell (i.e. it will not produce a list
      // of 'image' neighbours at further and further distances). Therefore
      // the r_cutoff in these tests must be less than the inradius of the
      // super cell.
      ASSERT_FALSE(pbc[0] && r_cutoff_ > jams::maths::parallelepiped_height(supercell_size_*b, supercell_size_*c, supercell_size_*a));
      ASSERT_FALSE(pbc[1] && r_cutoff_ > jams::maths::parallelepiped_height(supercell_size_*c, supercell_size_*a, supercell_size_*b));
      ASSERT_FALSE(pbc[2] && r_cutoff_ > jams::maths::parallelepiped_height(supercell_size_*a, supercell_size_*b, supercell_size_*c));

      near_tree_ = std::make_unique<jams::InteractionNearTree>(supercell_size*a, supercell_size*b, supercell_size*c, pbc, r_cutoff, epsilon);

      std::vector<Vec3> sites;
      int count = 0;
      for (auto i = 0; i < supercell_size_; ++i) {
        for (auto j = 0; j < supercell_size_; ++j) {
          for (auto k = 0; k < supercell_size_; ++k) {
            positions_.emplace_back(i * a + j * b + k * c, count);
            sites.push_back(i * a + j * b + k * c);
            count++;
          }
        }
      }

      near_tree_->insert_sites(sites);
    }


    void bruteforce_comparison_test() {
      for (const auto &r_i : positions_) {

        std::vector<Position> near_tree_neighbours = near_tree_->neighbours(r_i.first, r_cutoff_);

        for (auto &r_j : near_tree_neighbours) {
          r_j.first = r_i.first - r_j.first;
        }

        std::vector<Position> brute_force_neighbours;

        for (const auto &r_j : positions_) {
          auto r_ij = jams::minimum_image(supercell_size_*a_, supercell_size_*b_, supercell_size_*c_, pbc_, r_i.first,
                                          r_j.first, epsilon_);
          if (norm(r_ij) < epsilon_) continue;
          if (definately_greater_than(norm(r_ij), r_cutoff_, epsilon_)) continue;
          brute_force_neighbours.emplace_back(r_ij, r_j.second);
        }

        auto positions_are_equivalent = [&](const Position& a, const Position& b){
            return approximately_equal(a.first, b.first, epsilon_) && a.second == b.second;
        };

//        std::cout << "bruteforce neighbours:   " << brute_force_neighbours.size() << "\n";
//        std::cout << "near tree neighbours:    " << near_tree_neighbours.size() << "\n";

        EXPECT_EQ(near_tree_neighbours.size(), brute_force_neighbours.size());
        EXPECT_TRUE(std::is_permutation(near_tree_neighbours.begin(), near_tree_neighbours.end(), brute_force_neighbours.begin(), positions_are_equivalent));
      }
    }


    Vec3 a_;
    Vec3 b_;
    Vec3 c_;
    Vec3b pbc_;
    int supercell_size_;
    double r_cutoff_;
    double epsilon_;

    std::unique_ptr<jams::InteractionNearTree> near_tree_;
    std::vector<Position> positions_;

};

///
/// @test
/// Tests that the minimum_image_smith_method function gives the shortest r_ij
/// for results where |r_ij| is less than the inradius of the cell.
///

TEST_F(InteractionNeartreeTest, neighbours_no_pbc_simple) {
  using namespace testing;

  Vec3 a = {1.0, 0.0, 0.0};
  Vec3 b = {0.0, 1.0, 0.0};
  Vec3 c = {0.0, 0.0, 1.0};
  Vec3b pbc = {false, false, false};
  int superlattice_size = 5;
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  SetUp(a, b, c, pbc, superlattice_size, r_cutoff, epsilon);

  bruteforce_comparison_test();
}

TEST_F(InteractionNeartreeTest, neighbours_pbc_simple) {
  using namespace testing;

  Vec3 a = {1.0, 0.0, 0.0};
  Vec3 b = {0.0, 1.0, 0.0};
  Vec3 c = {0.0, 0.0, 1.0};
  Vec3b pbc = {true, true, true};
  int superlattice_size = 5;
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  SetUp(a, b, c, pbc, superlattice_size, r_cutoff, epsilon);

  bruteforce_comparison_test();
}

TEST_F(InteractionNeartreeTest, neighbours_no_pbc_complicated) {
  using namespace testing;

  Vec3 a = {0.5, 0.0, 0.0};
  Vec3 b = {-0.25, 0.4330127019, 0.0};
  Vec3 c = {0.0, 0.2886751346, 0.4082482905};
  Vec3b pbc = {false, false, false};
  int superlattice_size = 5;
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  SetUp(a, b, c, pbc, superlattice_size, r_cutoff, epsilon);

  bruteforce_comparison_test();
}

TEST_F(InteractionNeartreeTest, neighbours_pbc_complicated) {
  using namespace testing;

  Vec3 a = {0.5, 0.0, 0.0};
  Vec3 b = {-0.25, 0.4330127019, 0.0};
  Vec3 c = {0.0, 0.2886751346, 0.4082482905};
  Vec3b pbc = {true, true, true};
  int superlattice_size = 12;
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  SetUp(a, b, c, pbc, superlattice_size, r_cutoff, epsilon);

  bruteforce_comparison_test();
}

TEST(InteractionNeartreeStandaloneTest, insert_sites_replaces_existing_sites) {
  const Vec3 a = {1.0, 0.0, 0.0};
  const Vec3 b = {0.0, 1.0, 0.0};
  const Vec3 c = {0.0, 0.0, 1.0};
  const Vec3b pbc = {false, false, false};
  const double r_cutoff = 1.0;
  const double epsilon = 1e-9;

  jams::InteractionNearTree near_tree(a, b, c, pbc, r_cutoff, epsilon);

  near_tree.insert_sites({Vec3{0.0, 0.0, 0.0}});
  near_tree.insert_sites({Vec3{0.0, 0.0, 0.0}});

  EXPECT_EQ(near_tree.num_neighbours(Vec3{0.0, 0.0, 0.0}, 0.1), 0);
}

TEST_F(InteractionNeartreeTest, num_neighbours_non_site_is_not_negative) {
  Vec3 a = {1.0, 0.0, 0.0};
  Vec3 b = {0.0, 1.0, 0.0};
  Vec3 c = {0.0, 0.0, 1.0};
  Vec3b pbc = {false, false, false};
  int superlattice_size = 5;
  double r_cutoff = 2.0;
  const double epsilon = 1e-5;

  SetUp(a, b, c, pbc, superlattice_size, r_cutoff, epsilon);

  EXPECT_EQ(near_tree_->num_neighbours(Vec3{-100.0, -100.0, -100.0}, 0.1), 0);
}

TEST(InteractionNeartreeStandaloneTest, shell_validates_width_and_bounds) {
  const Vec3 a = {1.0, 0.0, 0.0};
  const Vec3 b = {0.0, 1.0, 0.0};
  const Vec3 c = {0.0, 0.0, 1.0};
  const Vec3b pbc = {true, true, true};
  const double r_cutoff = 1.0;
  const double epsilon = 1e-9;

  jams::InteractionNearTree near_tree(a, b, c, pbc, r_cutoff, epsilon);
  near_tree.insert_sites({Vec3{0.0, 0.0, 0.0}});

  EXPECT_THROW(near_tree.shell(Vec3{0.0, 0.0, 0.0}, 0.5, 0.0), std::invalid_argument);
  EXPECT_THROW(near_tree.shell(Vec3{0.0, 0.0, 0.0}, 0.95, 0.2), std::invalid_argument);
}



#endif //JAMS_LATTICE_INTERACTION_NEARTREE_TEST_H
