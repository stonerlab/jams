#ifndef JAMS_LATTICE_MINIMUM_IMAGE_TEST_H
#define JAMS_LATTICE_MINIMUM_IMAGE_TEST_H

#include <jams/lattice/minimum_image.h>
#include <jams/maths/parallelepiped.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

///
/// @test
/// Tests that the minimum_image_smith_method function gives the shortest r_ij
/// for results where |r_ij| is less than the inradius of the cell.
///
TEST(LatticeMinimumImageTest, minimum_image_smith_method) {
    using namespace testing;

    const double double_epsilon = 1e-9;

    Vec3b pbc = {true, true, true};

    auto random_vector = [&](){
        return Vec3 {rand()/double(RAND_MAX), rand()/double(RAND_MAX), rand()/double(RAND_MAX)};
    };

    int num_test_cells = 50;
    int num_test_positions = 100000;

    for (auto n = 0; n < num_test_cells; ++n) {
        Vec3 a = random_vector();
        Vec3 b = random_vector();
        Vec3 c = random_vector();

        auto random_position = [&]() {
            return a * rand() / double(RAND_MAX) +
                   b * rand() / double(RAND_MAX) +
                   c * rand() / double(RAND_MAX);
        };

        for (auto i = 0; i < num_test_positions; ++i) {
            const Vec3 r_i = random_position();
            const Vec3 r_j = random_position();

            Vec3 r_ij_smith, r_ij_brute;

            r_ij_smith = jams::minimum_image_smith_method(a, b, c, pbc, r_i, r_j);
          if (!definately_less_than(norm(r_ij_smith),
                                   jams::maths::parallelepiped_inradius(a, b, c))) {
                continue;
            }
            r_ij_brute = jams::minimum_image_bruteforce(a, b, c, pbc, r_i, r_j);

            const Vec<Matcher<double>, 3> result = {
                    DoubleNear(r_ij_brute[0], double_epsilon),
                    DoubleNear(r_ij_brute[1], double_epsilon),
                    DoubleNear(r_ij_brute[2], double_epsilon)
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
/// Testing this is difficult because it must already assume that our bruteforce
/// method works correctly. Here we avoid assuming that the bruteforce automated
/// selection of offset_depth is correct and compare with an explicit and large
/// offset_depth. We use randomly generated cells and positions because this
/// function should work for any possible cell and positions.
///
TEST(LatticeMinimumImageTest, minimum_image) {
    using namespace testing;

    const double double_epsilon = 1e-9;

    Vec3b pbc = {true, true, true};


    auto random_vector = [&](){
        return Vec3 {rand()/double(RAND_MAX), rand()/double(RAND_MAX), rand()/double(RAND_MAX)};
    };

    int num_test_cells = 20;
    int num_test_positions = 10;

    for (auto n = 0; n < num_test_cells; ++n) {
        Vec3 a = random_vector();
        Vec3 b = random_vector();
        Vec3 c = random_vector();

        auto random_position = [&](){
            return a * rand()/double(RAND_MAX) + b * rand()/double(RAND_MAX) + c * rand()/double(RAND_MAX);
        };

        for (auto i = 0; i < num_test_positions; ++i) {
            const Vec3 r_i = random_position();
            const Vec3 r_j = random_position();

            auto r_ij = jams::minimum_image(a, b, c, pbc, r_i, r_j);

            auto r_ij_brute = jams::minimum_image_bruteforce_explicit_depth(a, b, c, pbc, r_i, r_j, {50, 50, 50});

            const Vec<Matcher<double>, 3> result = {
                    DoubleNear(r_ij_brute[0], double_epsilon),
                    DoubleNear(r_ij_brute[1], double_epsilon),
                    DoubleNear(r_ij_brute[2], double_epsilon)
            };

            ASSERT_THAT(r_ij, ElementsAreArray(result)) << "|r_ij|: " << norm(r_ij) << "|r_ij_ref|: " << norm(r_ij_brute);
        }
    }
}

#endif //JAMS_LATTICE_MINIMUM_IMAGE_TEST_H