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

#if HAS_CUDA
namespace {
bool have_multiarray_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
}  // namespace
#endif

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

TEST(MultiArrayDetailsRuntimeTest, IndicesInBounds) {
  using jams::detail::indices_in_bounds;

  const std::array<std::size_t, 3> dims{2, 3, 4};

  EXPECT_TRUE(indices_in_bounds(dims, 1u, 2u, 3u));
  EXPECT_TRUE(indices_in_bounds(dims, std::array<std::size_t, 3>{1, 0, 0}));
  EXPECT_FALSE(indices_in_bounds(dims, 2u, 2u, 3u));
  EXPECT_FALSE(indices_in_bounds(dims, std::array<std::size_t, 3>{1, 3, 0}));
}


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

TEST(MultiArrayIteratorCtorTest, SupportsForwardIteratorsForOneDimensionalArrays) {
  const std::vector<int> source{5, 6, 7, 8};

  jams::MultiArray<int, 1> values(source.begin(), source.end());

  ASSERT_EQ(values.size(), source.size());
  EXPECT_TRUE(std::equal(values.begin(), values.end(), source.begin(), source.end()));
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

TEST(MultiArrayOneDimensionalTest, SizeByAxisMatchesSize) {
  jams::MultiArray<int, 1> values(7);

  ASSERT_EQ(values.size(), 7u);
  ASSERT_EQ(values.size(0), 7u);
}

TEST(MultiArrayEmptyContainerTest, EmptyIteratorsAndFillAreSafe) {
  jams::MultiArray<int, 2> grid;
  const auto& const_grid = static_cast<const jams::MultiArray<int, 2>&>(grid);
  EXPECT_EQ(grid.begin(), grid.end());
  EXPECT_EQ(const_grid.begin(), const_grid.end());

  grid.fill(7);
  EXPECT_TRUE(grid.empty());
  EXPECT_EQ(grid.begin(), grid.end());

  jams::MultiArray<int, 1> line;
  EXPECT_EQ(line.begin(), line.end());
  line.fill(9);
  EXPECT_TRUE(line.empty());
  EXPECT_EQ(line.begin(), line.end());
}

TEST(MultiArrayIndexTypeTest, SupportsNonDefaultIndexTypes) {
  jams::MultiArray<int, 2, int> values(std::array<short, 2>{2, 3});

  ASSERT_EQ(values.size(0), 2);
  ASSERT_EQ(values.size(1), 3);
  ASSERT_EQ(values.elements(), 6);

  values(1, 2) = 7;
  EXPECT_EQ(values(1, 2), 7);

  values.resize(std::array<int, 2>{4, 5});
  ASSERT_EQ(values.size(0), 4);
  ASSERT_EQ(values.size(1), 5);
  ASSERT_EQ(values.elements(), 20);

  values(std::array<int, 2>{3, 4}) = 11;
  EXPECT_EQ(values(std::array<int, 2>{3, 4}), 11);
}

TEST(MultiArrayReadOnlyAccessTest, ProvidesReadOnlyHostIterationOnMutableArray) {
  jams::MultiArray<int, 2> values(2, 2);
  values.fill(3);

  const auto* data = values.read_only_data();
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(values.read_only_begin(), data);
  EXPECT_EQ(values.read_only_end(), data + values.elements());
  EXPECT_EQ(data[0], 3);
  EXPECT_EQ(data[3], 3);
}

TEST(MultiArrayResizeTest, UpdatesShapeAndElementCount) {
  jams::MultiArray<int, 2> values(2, 3);
  ASSERT_EQ(values.elements(), 6u);

  values.resize(4, 5);
  ASSERT_EQ(values.size(0), 4u);
  ASSERT_EQ(values.size(1), 5u);
  ASSERT_EQ(values.elements(), 20u);

  values.resize(std::array<int, 2>{3, 6});
  ASSERT_EQ(values.size(0), 3u);
  ASSERT_EQ(values.size(1), 6u);
  ASSERT_EQ(values.elements(), 18u);
}

TEST(MultiArrayAlgorithmsTest, ElementSumAddsMatchingShapes) {
  jams::MultiArray<int, 2> lhs(2, 2);
  jams::MultiArray<int, 2> rhs(2, 2);

  lhs(0, 0) = 1;
  lhs(0, 1) = 2;
  lhs(1, 0) = 3;
  lhs(1, 1) = 4;

  rhs(0, 0) = 10;
  rhs(0, 1) = 20;
  rhs(1, 0) = 30;
  rhs(1, 1) = 40;

  jams::element_sum(lhs, rhs);

  EXPECT_EQ(lhs(0, 0), 11);
  EXPECT_EQ(lhs(0, 1), 22);
  EXPECT_EQ(lhs(1, 0), 33);
  EXPECT_EQ(lhs(1, 1), 44);
}

#if HAS_CUDA
TEST(MultiArrayReadOnlyAccessTest, ReadOnlyHostAccessDoesNotDirtyDeviceState) {
  if (!have_multiarray_cuda_device()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  jams::MultiArray<int, 1> values(3, 0);
  const int device_values[3] = {1, 2, 3};

  ASSERT_NE(values.device_data(), nullptr);
  ASSERT_EQ(cudaMemcpy(values.device_data(), device_values, values.bytes(), cudaMemcpyHostToDevice), cudaSuccess);

  const int* host = values.read_only_data();
  ASSERT_NE(host, nullptr);
  EXPECT_EQ(host[0], 1);
  EXPECT_EQ(host[1], 2);
  EXPECT_EQ(host[2], 3);

  // Deliberately mutate the host storage behind the const API so the test can
  // detect whether read_only_data() accidentally marked the host copy dirty.
  auto* writable_host = const_cast<int*>(host);
  writable_host[0] = 9;

  int device_snapshot[3] = {0, 0, 0};
  ASSERT_EQ(cudaMemcpy(device_snapshot,
                       values.read_only_device_data(),
                       values.bytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(device_snapshot[0], 1);
  EXPECT_EQ(device_snapshot[1], 2);
  EXPECT_EQ(device_snapshot[2], 3);
}
#endif

#endif //JAMS_TEST_MULTIARRAY_H
