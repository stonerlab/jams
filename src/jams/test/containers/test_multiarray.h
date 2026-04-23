//
// Created by Joseph Barker on 2019-05-10.
//

#ifndef JAMS_TEST_MULTIARRAY_H
#define JAMS_TEST_MULTIARRAY_H

#include <array>
#include <complex>
#include <iterator>
#include <sstream>
#include <utility>

#include "jams/containers/multiarray.h"

// fixture class
template <typename T>
class MultiArrayDetailsTest : public testing::Test {

};

typedef testing::Types<int, unsigned, long, unsigned long> IntTypes;
TYPED_TEST_SUITE(MultiArrayDetailsTest, IntTypes);


// test array cast
TYPED_TEST(MultiArrayDetailsTest, array_cast) {
  using namespace jams;
  using namespace testing;
  auto x = detail::array_cast<TypeParam>(std::array<int,3>{1,2,3});
  ASSERT_THAT(x, ElementsAre(1,2,3));
}

// test product
TYPED_TEST(MultiArrayDetailsTest, product) {
  using namespace jams;

  ASSERT_EQ(detail::product(5), 5);
  ASSERT_EQ(detail::product(5,4,3,2,1), 120);
  auto x = detail::vec<TypeParam,1,1>::last_n_product(std::array<TypeParam, 1>{5});
  ASSERT_EQ(x, 5);
  auto y = detail::vec<TypeParam,5,5>::last_n_product(std::array<TypeParam, 5>{5, 4, 3, 2, 1});
  ASSERT_EQ(y, 120);
}

// test row major


// fixture class
template <typename T>
class MultiArrayTest : public testing::Test {

};

typedef testing::Types<int, unsigned, double, std::complex<double>, std::array<int,3>, std::array<double,3>> MultiArrayTypes;
TYPED_TEST_SUITE(MultiArrayTest, MultiArrayTypes);


TYPED_TEST(MultiArrayTest, ctor) {
  using namespace jams;

  MultiArray<TypeParam, 1> x1;
  ASSERT_EQ(x1.size(), 0);

  MultiArray<TypeParam, 2> x2;
  ASSERT_EQ(x2.size(0), 0);
  ASSERT_EQ(x2.size(1), 0);

  MultiArray<TypeParam, 1> y1(10);
  ASSERT_EQ(y1.size(), 10);

  MultiArray<TypeParam, 2> y2(10,3);
  ASSERT_EQ(y2.size(0), 10);
  ASSERT_EQ(y2.size(1), 3);

  MultiArray<TypeParam, 1> z1(std::array<typename MultiArray<TypeParam, 1>::size_type,1>{10});
  ASSERT_EQ(z1.size(), 10);

  MultiArray<TypeParam, 2> z2(std::array<typename MultiArray<TypeParam, 1>::size_type,2>{10,3});
  ASSERT_EQ(z2.size(0), 10);
  ASSERT_EQ(z2.size(1), 3);

  // check the arrays convert from other int types to size_type
  MultiArray<TypeParam, 1> a1(std::array<int,1>{10});
  ASSERT_EQ(a1.size(), 10);

  MultiArray<TypeParam, 2> a2(std::array<int,2>{10,3});
  ASSERT_EQ(a2.size(0), 10);
  ASSERT_EQ(a2.size(1), 3);

  MultiArray<TypeParam, 2> a3(10, 3);
  ASSERT_EQ(a3.size(0), 10);
  ASSERT_EQ(a3.size(1), 3);
  
}

TEST(MultiArrayNoexceptTest, AccessorsCanThrow) {
  using Array1 = jams::MultiArray<int, 1>;
  using Array2 = jams::MultiArray<int, 2>;

  EXPECT_FALSE(noexcept(std::declval<Array1&>().data()));
  EXPECT_FALSE(noexcept(std::declval<const Array1&>().data()));
  EXPECT_FALSE(noexcept(std::declval<Array2&>().device_data()));
  EXPECT_FALSE(noexcept(std::declval<const Array2&>().begin()));
  EXPECT_FALSE(noexcept(std::declval<const Array2&>().end()));
  EXPECT_FALSE(noexcept(std::declval<Array2&>().zero()));
}

TEST(MultiArrayIteratorCtorTest, SupportsInputIteratorsForOneDimensionalArrays) {
  std::istringstream input("1 2 3 4");
  std::istream_iterator<int> first(input);
  std::istream_iterator<int> last;

  jams::MultiArray<int, 1> values(first, last);

  ASSERT_EQ(values.size(), 4u);
  EXPECT_EQ(values(0), 1);
  EXPECT_EQ(values(1), 2);
  EXPECT_EQ(values(2), 3);
  EXPECT_EQ(values(3), 4);
}

TEST(MultiArrayExtentValidationTest, RejectsNegativeExtents) {
  EXPECT_THROW((jams::MultiArray<int, 2>(-1, 3)), std::length_error);
  EXPECT_THROW((jams::MultiArray<int, 2>(std::array<int, 2>{2, -1})), std::length_error);

  jams::MultiArray<int, 2> grid(2, 3);
  EXPECT_THROW(grid.resize(2, -1), std::length_error);

  EXPECT_THROW((jams::MultiArray<int, 1, int>(-1)), std::length_error);

  jams::MultiArray<int, 1, int> line(2);
  EXPECT_THROW(line.resize(-1), std::length_error);
}

#endif //JAMS_TEST_MULTIARRAY_H
