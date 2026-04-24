//
// Created by Joseph Barker on 2019-05-10.
//

#ifndef JAMS_TEST_MULTIARRAY_H
#define JAMS_TEST_MULTIARRAY_H

#include <array>
#include <complex>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <vector>

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

TEST(MultiArrayFinalApiTest, ConstAccessorsAreLogicallyConstAndMaySynchronize) {
  using namespace jams;
  using Array = MultiArray<int, 1>;

  static_assert(std::is_same<decltype(std::declval<Array&>().data()), int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<const Array&>().data()), const int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<Array&>().device_data()), int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<const Array&>().device_data()), const int*>::value, "");
  static_assert(!noexcept(std::declval<Array&>().data()), "");
  static_assert(!noexcept(std::declval<const Array&>().data()), "");
  static_assert(!noexcept(std::declval<Array&>().device_data()), "");
  static_assert(!noexcept(std::declval<const Array&>().device_data()), "");
  static_assert(!noexcept(std::declval<Array&>().begin()), "");
  static_assert(!noexcept(std::declval<const Array&>().begin()), "");
  static_assert(!noexcept(std::declval<Array&>().zero()), "");

  Array values(4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values(i) = static_cast<int>(i + 1);
  }

  const Array& const_values = values;
  const int* data = const_values.data();
  ASSERT_NE(data, nullptr);
  EXPECT_THAT(std::vector<int>(data, data + const_values.size()),
              testing::ElementsAre(1, 2, 3, 4));
  EXPECT_EQ(const_values(2), 3);
}

TEST(MultiArrayFinalApiTest, ConstAccessorsWorkForMultidimensionalArrays) {
  using namespace jams;

  MultiArray<int, 2> values(2, 3);
  int n = 0;
  for (std::size_t i = 0; i < values.size(0); ++i) {
    for (std::size_t j = 0; j < values.size(1); ++j) {
      values(i, j) = ++n;
    }
  }

  const MultiArray<int, 2>& const_values = values;
  EXPECT_EQ(const_values.shape(), (std::array<MultiArray<int, 2>::size_type, 2>{2, 3}));
  EXPECT_EQ(const_values(0, 0), 1);
  EXPECT_EQ(const_values(0, 2), 3);
  EXPECT_EQ(const_values(1, 0), 4);
  EXPECT_EQ(const_values(1, 2), 6);

  const int* data = const_values.data();
  ASSERT_NE(data, nullptr);
  EXPECT_THAT(std::vector<int>(data, data + const_values.elements()),
              testing::ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(MultiArrayFinalApiTest, OneDimensionalInputIteratorConstructionConsumesSinglePassRangeOnce) {
  using namespace jams;

  std::istringstream input("9 2 6 5");
  std::istream_iterator<int> first(input);
  std::istream_iterator<int> last;

  MultiArray<int, 1> values(first, last);
  EXPECT_EQ(values.size(), 4u);
  EXPECT_EQ(values.elements(), 4u);

  const int* data = values.data();
  ASSERT_NE(data, nullptr);
  EXPECT_THAT(std::vector<int>(data, data + values.size()),
              testing::ElementsAre(9, 2, 6, 5));
}

TEST(MultiArrayFinalApiTest, CopyMoveAndSwapPreserveLogicalValues) {
  using namespace jams;
  using testing::ElementsAre;

  MultiArray<int, 1> source(3);
  source(0) = 7;
  source(1) = 8;
  source(2) = 9;

  MultiArray<int, 1> copy(source);
  source(0) = -1;
  EXPECT_THAT(std::vector<int>(copy.data(), copy.data() + copy.size()), ElementsAre(7, 8, 9));

  MultiArray<int, 1> assigned(1);
  assigned = copy;
  copy(1) = -2;
  EXPECT_THAT(std::vector<int>(assigned.data(), assigned.data() + assigned.size()), ElementsAre(7, 8, 9));

  MultiArray<int, 1> moved(std::move(assigned));
  EXPECT_EQ(moved.size(), 3u);
  EXPECT_THAT(std::vector<int>(moved.data(), moved.data() + moved.size()), ElementsAre(7, 8, 9));

  MultiArray<int, 1> other(2);
  other(0) = 1;
  other(1) = 2;
  swap(moved, other);
  EXPECT_THAT(std::vector<int>(moved.data(), moved.data() + moved.size()), ElementsAre(1, 2));
  EXPECT_THAT(std::vector<int>(other.data(), other.data() + other.size()), ElementsAre(7, 8, 9));
}

#endif //JAMS_TEST_MULTIARRAY_H
