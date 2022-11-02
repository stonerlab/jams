#ifndef JAMS_LATTICE_MINIMUM_IMAGE_TEST_H
#define JAMS_LATTICE_MINIMUM_IMAGE_TEST_H

#include <jams/lattice/minimum_image.h>
#include <jams/maths/parallelepiped.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <jams/helpers/random.h>

class LatticeMinimumImageTest : public ::testing::TestWithParam<std::tuple<Vec3, Vec3, Vec3>> {
protected:

    // These positions are in fractional coordinates (they are multiples of
    // the lattice vectors a,b,c NOT cartesian points within the unit cell).
    //
    // We include a selection of positions which cover lots of possible edge
    // cases, e.g. the point (0.5, 0.5, 0.5) is generate for cubic cells.
    std::vector<Vec3> positions = {
        // origin
        Vec3{0.0, 0.0, 0.0},

        // body center
        Vec3{0.5, 0.5, 0.5},

        // edges
        Vec3{0.25, 0.0, 0.0},
        Vec3{0.0, 0.25, 0.0},
        Vec3{0.0, 0.0, 0.25},

        Vec3{0.5, 0.0, 0.0},
        Vec3{0.0, 0.5, 0.0},
        Vec3{0.0, 0.0, 0.5},

        Vec3{0.75, 0.0, 0.0},
        Vec3{0.0, 0.75, 0.0},
        Vec3{0.0, 0.0, 0.75},

        // faces
        Vec3{0.0, 0.25, 0.25},
        Vec3{0.25, 0.0, 0.25},
        Vec3{0.25, 0.25, 0.0},
        Vec3{0.25, 0.25, 0.0},

        Vec3{0.0, 0.5, 0.5},
        Vec3{0.5, 0.0, 0.5},
        Vec3{0.5, 0.5, 0.0},
        Vec3{0.5, 0.5, 0.0},

        Vec3{0.0, 0.75, 0.75},
        Vec3{0.75, 0.0, 0.75},
        Vec3{0.75, 0.75, 0.0},
        Vec3{0.75, 0.75, 0.0},

        // quadrants
        Vec3{0.25, 0.25, 0.25},
        Vec3{0.25, 0.25, 0.75},
        Vec3{0.25, 0.75, 0.25},
        Vec3{0.75, 0.25, 0.25},
        Vec3{0.75, 0.75, 0.25},
        Vec3{0.75, 0.25, 0.75},
        Vec3{0.25, 0.75, 0.75},
        Vec3{0.75, 0.75, 0.75},

        // thirds
        Vec3{1.0/3.0, 1.0/3.0, 1.0/3.0},
        Vec3{2.0/3.0, 2.0/3.0, 2.0/3.0},
    };
};

///
/// @test
/// Tests that the minimum_image_smith_method function gives the shortest r_ij
/// in the region of validity of the algorithm where |r_ij| is less than the
/// inradius of the cell.
///
TEST_P(LatticeMinimumImageTest, minimum_image_smith_method) {
  const double double_epsilon = 1e-6;

  Vec3 a = std::get<0>(GetParam());
  Vec3 b = std::get<1>(GetParam());
  Vec3 c = std::get<2>(GetParam());

  Vec3b pbc = {true, true, true};

  for (auto i = 0; i < positions.size(); ++i) {
    for (auto j = 0; j < positions.size(); ++j) {
      const Vec3 r_i = a * positions[i][0] + b * positions[i][1] + c * positions[i][2];
      const Vec3 r_j = a * positions[j][0] + b * positions[j][1] + c * positions[j][2];

      Vec3 r_ij_smith, r_ij_brute;

      r_ij_smith = jams::minimum_image_smith_method(a, b, c, pbc, r_i, r_j);
      if (!definately_less_than(norm(r_ij_smith),
                                jams::maths::parallelepiped_inradius(a, b, c),
                                double_epsilon)) {
        continue;
      }

      r_ij_brute = jams::minimum_image_bruteforce(a, b, c, pbc, r_i, r_j,
                                                  double_epsilon);

      const Vec<::testing::Matcher<double>, 3> result = {
          ::testing::DoubleNear(r_ij_brute[0], double_epsilon),
          ::testing::DoubleNear(r_ij_brute[1], double_epsilon),
          ::testing::DoubleNear(r_ij_brute[2], double_epsilon)
      };

      EXPECT_THAT(r_ij_smith, ElementsAreArray(result));
    }
  }
}

///
/// @test
/// Tests that the general minimum_image function which should work in all cases
/// gives the shortest r_ij.
///
/// @attention
/// We have a stricter requirement than just r_ij being the shortest, but also
/// r_ij from the minimum_image function should also give the same *vector* as
/// the bruteforce method (i.e. degenerate distances should consistently
/// select the same vector amongst the possible choices).
///
/// @warning
/// This test already assumes that our bruteforce method works correctly. Here
/// we avoid assuming that the bruteforce automated selection of offset_depth is
/// correct and compare with an explicit and large offset_depth. This simpler
/// bruteforce implementation is very simple code *likely* to be bug free, but
/// means this test is not suitable for very skew cells where the offset_depth
/// may be insufficient.
///
TEST_P(LatticeMinimumImageTest, minimum_image) {
  const double double_epsilon = 1e-6;

  Vec3 a = std::get<0>(GetParam());
  Vec3 b = std::get<1>(GetParam());
  Vec3 c = std::get<2>(GetParam());

  Vec3b pbc = {true, true, true};

  for (auto i = 0; i < positions.size(); ++i) {
    for (auto j = 0; j < positions.size(); ++j) {
      const Vec3 r_i = a * positions[i][0] + b * positions[i][1] + c * positions[i][2];
      const Vec3 r_j = a * positions[j][0] + b * positions[j][1] + c * positions[j][2];

      auto r_ij = jams::minimum_image(a, b, c, pbc, r_i, r_j, double_epsilon);
      auto r_ij_brute = jams::minimum_image_bruteforce_explicit_depth(a, b, c, pbc, r_i, r_j, {9, 9, 9}, double_epsilon);

      const Vec<testing::Matcher<double>, 3> result = {
          testing::DoubleNear(r_ij_brute[0], double_epsilon),
          testing::DoubleNear(r_ij_brute[1], double_epsilon),
          testing::DoubleNear(r_ij_brute[2], double_epsilon)
      };

      EXPECT_THAT(r_ij, ElementsAreArray(result)) << "|r_ij_test|: " << norm(r_ij) << "|r_ij_bruteforce|: " << norm(r_ij_brute);
    }
  }
}

// A selection of Bravais lattices to test against. In principle we could
// generate random lattice vectors (and we used to) but very skew cells take
// a very long time to test with bruteforce. If using random cells we can't use
// an explicit depth search but have to calculate the depth too otherwise we
// had random failures when the cell was too skew for the current depth.
// Any cells which appear to break the minimum image code should be added here
// as test cases.
INSTANTIATE_TEST_SUITE_P(LatticeTypes, LatticeMinimumImageTest,
                         testing::Values(
                             std::tuple<Vec3, Vec3, Vec3>({1,0,0}, {0,1,0}, {0,0,1}),             // cubic
                             std::tuple<Vec3, Vec3, Vec3>({1,0,0}, {0.5,sqrt(3)/2,0}, {0,0,1.2}), // hexagonal
                             std::tuple<Vec3, Vec3, Vec3>({0.5, -1/(2*sqrt(3)), 1.0/3.0 }, {1, 1/sqrt(3), 1.0/3.0}, {-0.5, -1/(2*sqrt(3)), 1.0/3.0}), // rhombohedral
                             std::tuple<Vec3, Vec3, Vec3>({1,0,0}, {0,1,0}, {0,0,0.98}),             // tetragonal
                             std::tuple<Vec3, Vec3, Vec3>({1,0,0}, {0,0.75,0}, {0,0,1.2})             // orthorhombic
                         ));

#endif //JAMS_LATTICE_MINIMUM_IMAGE_TEST_H
