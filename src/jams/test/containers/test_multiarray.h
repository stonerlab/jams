//
// Created by Joseph Barker on 2019-05-10.
//

#ifndef JAMS_TEST_MULTIARRAY_H
#define JAMS_TEST_MULTIARRAY_H

#include <array>
#include <complex>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "jams/containers/multiarray.h"

#if HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {
bool multiarray_cuda_device_available() {
#if HAS_CUDA
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}
}

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
  ASSERT_EQ(x2.extent(0), 0);
  ASSERT_EQ(x2.extent(1), 0);

  MultiArray<TypeParam, 1> y1(10);
  ASSERT_EQ(y1.size(), 10);

  MultiArray<TypeParam, 2> y2(10,3);
  ASSERT_EQ(y2.extent(0), 10);
  ASSERT_EQ(y2.extent(1), 3);

  MultiArray<TypeParam, 1> z1(std::array<typename MultiArray<TypeParam, 1>::size_type,1>{10});
  ASSERT_EQ(z1.size(), 10);

  MultiArray<TypeParam, 2> z2(std::array<typename MultiArray<TypeParam, 1>::size_type,2>{10,3});
  ASSERT_EQ(z2.extent(0), 10);
  ASSERT_EQ(z2.extent(1), 3);

  // check the arrays convert from other int types to size_type
  MultiArray<TypeParam, 1> a1(std::array<int,1>{10});
  ASSERT_EQ(a1.size(), 10);

  MultiArray<TypeParam, 2> a2(std::array<int,2>{10,3});
  ASSERT_EQ(a2.extent(0), 10);
  ASSERT_EQ(a2.extent(1), 3);

  MultiArray<TypeParam, 2> a3(10, 3);
  ASSERT_EQ(a3.extent(0), 10);
  ASSERT_EQ(a3.extent(1), 3);
  
}

TEST(MultiArrayFinalApiTest, ConstAccessorsAreLogicallyConstAndMaySynchronize) {
  using namespace jams;
  using Array = MultiArray<int, 1>;

  static_assert(std::is_same<decltype(std::declval<Array&>().data()), int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<const Array&>().data()), const int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<Array&>().host_data()), int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<const Array&>().host_data()), const int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<Array&>().device_data()), int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<const Array&>().device_data()), const int*>::value, "");
  static_assert(std::is_same<decltype(std::declval<Array&>().mutable_device_data()), int*>::value, "");
  static_assert(std::is_nothrow_default_constructible<Array>::value, "");
  static_assert(std::is_nothrow_move_constructible<Array>::value, "");
  static_assert(std::is_nothrow_move_assignable<Array>::value, "");
  static_assert(noexcept(std::declval<Array&>().clear()), "");
  static_assert(noexcept(std::declval<Array&>().swap(std::declval<Array&>())), "");
  static_assert(noexcept(swap(std::declval<Array&>(), std::declval<Array&>())), "");
  static_assert(noexcept(std::declval<const Array&>().empty()), "");
  static_assert(noexcept(std::declval<const Array&>().size()), "");
  static_assert(noexcept(std::declval<const Array&>().extent(0)), "");
  static_assert(noexcept(std::declval<const Array&>().shape()), "");
  static_assert(!noexcept(std::declval<const Array&>().bytes()), "");
  static_assert(noexcept(std::declval<const Array&>().elements()), "");
  static_assert(noexcept(std::declval<const Array&>().rank()), "");
  static_assert(!noexcept(std::declval<const Array&>().max_size()), "");
  static_assert(!noexcept(std::declval<Array&>().data()), "");
  static_assert(!noexcept(std::declval<const Array&>().data()), "");
  static_assert(!noexcept(std::declval<Array&>().host_data()), "");
  static_assert(!noexcept(std::declval<const Array&>().host_data()), "");
  static_assert(!noexcept(std::declval<Array&>().device_data()), "");
  static_assert(!noexcept(std::declval<const Array&>().device_data()), "");
  static_assert(!noexcept(std::declval<Array&>().mutable_device_data()), "");
  static_assert(!noexcept(std::declval<Array&>().begin()), "");
  static_assert(!noexcept(std::declval<const Array&>().begin()), "");
  static_assert(!noexcept(std::declval<Array&>().end()), "");
  static_assert(!noexcept(std::declval<const Array&>().end()), "");
  static_assert(!noexcept(std::declval<Array&>().zero()), "");
  static_assert(noexcept(std::declval<Array&>().release_stale_host()), "");
  static_assert(noexcept(std::declval<Array&>().release_stale_device()), "");

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

TEST(MultiArrayFinalApiTest, UniformCapacityApiForOneAndTwoDimensions) {
  using namespace jams;

  MultiArray<int, 1> line(4);
  EXPECT_FALSE(line.empty());
  EXPECT_EQ(line.rank(), 1u);
  EXPECT_EQ(line.size(), 4u);
  EXPECT_EQ(line.elements(), 4u);
  EXPECT_EQ(line.extent(0), 4u);
  EXPECT_EQ(line.shape(), (std::array<MultiArray<int, 1>::size_type, 1>{4}));

  MultiArray<int, 2> grid(2, 3);
  EXPECT_FALSE(grid.empty());
  EXPECT_EQ(grid.rank(), 2u);
  EXPECT_EQ(grid.size(), 6u);
  EXPECT_EQ(grid.elements(), 6u);
  EXPECT_EQ(grid.extent(0), 2u);
  EXPECT_EQ(grid.extent(1), 3u);
  EXPECT_EQ(grid.shape(), (std::array<MultiArray<int, 2>::size_type, 2>{2, 3}));
}

TEST(MultiArrayFinalApiTest, ConstAccessorsWorkForMultidimensionalArrays) {
  using namespace jams;

  MultiArray<int, 2> values(2, 3);
  int n = 0;
  for (std::size_t i = 0; i < values.extent(0); ++i) {
    for (std::size_t j = 0; j < values.extent(1); ++j) {
      values(i, j) = ++n;
    }
  }

  const MultiArray<int, 2>& const_values = values;
  EXPECT_EQ(const_values.shape(), (std::array<MultiArray<int, 2>::size_type, 2>{2, 3}));
  EXPECT_EQ(const_values.rank(), 2u);
  EXPECT_EQ(const_values.size(), 6u);
  EXPECT_EQ(const_values.extent(0), 2u);
  EXPECT_EQ(const_values.extent(1), 3u);
  EXPECT_EQ(const_values(0, 0), 1);
  EXPECT_EQ(const_values(0, 2), 3);
  EXPECT_EQ(const_values(1, 0), 4);
  EXPECT_EQ(const_values(1, 2), 6);

  const int* data = const_values.data();
  ASSERT_NE(data, nullptr);
  EXPECT_THAT(std::vector<int>(data, data + const_values.elements()),
              testing::ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(MultiArrayFinalApiTest, OneDimensionalAndMultidimensionalIndexingUseUniformCallSyntax) {
  using namespace jams;

  MultiArray<int, 1> line(3);
  line(0) = 11;
  line(1) = 12;
  line(2) = 13;
  EXPECT_EQ(line(std::array<MultiArray<int, 1>::size_type, 1>{0}), 11);
  EXPECT_EQ(line(std::array<MultiArray<int, 1>::size_type, 1>{2}), 13);

  MultiArray<int, 3> cube(2, 2, 2);
  cube(0, 0, 0) = 1;
  cube(0, 1, 1) = 4;
  cube(1, 0, 0) = 5;
  cube(1, 1, 1) = 8;
  EXPECT_EQ(cube(std::array<MultiArray<int, 3>::size_type, 3>{0, 1, 1}), 4);
  EXPECT_EQ(cube(std::array<MultiArray<int, 3>::size_type, 3>{1, 1, 1}), 8);
}

TEST(MultiArrayFinalApiTest, OneDimensionalInputIteratorConstructionConsumesSinglePassRangeOnce) {
  using namespace jams;

  std::istringstream input("9 2 6 5");
  std::istream_iterator<int> first(input);
  std::istream_iterator<int> last;

  MultiArray<int, 1> values(first, last);
  EXPECT_EQ(values.size(), 4u);
  EXPECT_EQ(values.extent(0), 4u);
  EXPECT_EQ(values.elements(), 4u);

  const int* data = values.data();
  ASSERT_NE(data, nullptr);
  EXPECT_THAT(std::vector<int>(data, data + values.size()),
              testing::ElementsAre(9, 2, 6, 5));
}

TEST(MultiArrayFinalApiTest, OneDimensionalForwardIteratorConstructionUsesExactLogicalSize) {
  using namespace jams;

  const std::vector<int> source = {4, 8, 15, 16, 23, 42};
  MultiArray<int, 1> values(source.begin(), source.end());
  EXPECT_EQ(values.size(), source.size());
  EXPECT_EQ(values.extent(0), source.size());
  EXPECT_EQ(values.elements(), source.size());
  EXPECT_EQ(values.bytes(), source.size() * sizeof(int));

  const int* data = values.data();
  ASSERT_NE(data, nullptr);
  EXPECT_THAT(std::vector<int>(data, data + values.size()),
              testing::ElementsAre(4, 8, 15, 16, 23, 42));
}

TEST(MultiArrayFinalApiTest, OneDimensionalRandomAccessIteratorConstructionUsesExactLogicalSize) {
  using namespace jams;

  const int source[] = {10, 20, 30};
  MultiArray<int, 1> values(std::begin(source), std::end(source));
  EXPECT_EQ(values.size(), 3u);
  EXPECT_EQ(values.extent(0), 3u);
  EXPECT_EQ(values.elements(), 3u);
  EXPECT_EQ(values.bytes(), sizeof(source));

  const int* data = values.data();
  ASSERT_NE(data, nullptr);
  EXPECT_THAT(std::vector<int>(data, data + values.size()),
              testing::ElementsAre(10, 20, 30));
}

TEST(MultiArrayFinalApiTest, OneDimensionalIntegralConstructionDoesNotSelectIteratorOverload) {
  using namespace jams;

  static_assert(std::is_constructible<MultiArray<int, 1>, MultiArray<int, 1>::size_type>::value, "");
  static_assert(std::is_constructible<MultiArray<int, 1>, int, MultiArray<int, 1>::size_type>::value, "");

  MultiArray<int, 1> sized(4);
  EXPECT_EQ(sized.size(), 4u);
  EXPECT_EQ(sized.extent(0), 4u);
  EXPECT_EQ(sized.elements(), 4u);

  MultiArray<int, 1> filled(12, 3);
  EXPECT_EQ(filled.size(), 3u);
  EXPECT_EQ(filled.extent(0), 3u);
  EXPECT_EQ(filled.elements(), 3u);
  EXPECT_THAT(std::vector<int>(filled.data(), filled.data() + filled.size()),
              testing::ElementsAre(12, 12, 12));
}

TEST(MultiArrayFinalApiTest, ResizeUsesUniformExtentAndTotalSizeApi) {
  using namespace jams;

  MultiArray<int, 1> line;
  line.resize(5);
  EXPECT_EQ(line.rank(), 1u);
  EXPECT_EQ(line.size(), 5u);
  EXPECT_EQ(line.extent(0), 5u);
  EXPECT_EQ(line.elements(), 5u);

  line.resize(std::array<MultiArray<int, 1>::size_type, 1>{2});
  EXPECT_EQ(line.size(), 2u);
  EXPECT_EQ(line.extent(0), 2u);
  EXPECT_EQ(line.elements(), 2u);

  MultiArray<int, 2> grid;
  grid.resize(3, 4);
  EXPECT_EQ(grid.rank(), 2u);
  EXPECT_EQ(grid.size(), 12u);
  EXPECT_EQ(grid.extent(0), 3u);
  EXPECT_EQ(grid.extent(1), 4u);
  EXPECT_EQ(grid.elements(), 12u);

  grid.resize(std::array<MultiArray<int, 2>::size_type, 2>{2, 5});
  EXPECT_EQ(grid.size(), 10u);
  EXPECT_EQ(grid.extent(0), 2u);
  EXPECT_EQ(grid.extent(1), 5u);
  EXPECT_EQ(grid.elements(), 10u);
}

TEST(MultiArrayFinalApiTest, ShapeConstructionAndResizeCheckExtentOverflow) {
  using namespace jams;

  EXPECT_THROW((MultiArray<int, 1, int>(-1)), std::length_error);
  EXPECT_THROW((MultiArray<int, 1, unsigned char>(std::array<unsigned, 1>{300u})), std::overflow_error);
  EXPECT_THROW((MultiArray<int, 2>(std::numeric_limits<std::size_t>::max(), std::size_t{2})),
               std::overflow_error);

  MultiArray<int, 2> values(2, 3);
  EXPECT_THROW(values.resize(std::numeric_limits<std::size_t>::max(), std::size_t{2}),
               std::overflow_error);
  EXPECT_EQ(values.shape(), (std::array<MultiArray<int, 2>::size_type, 2>{2, 3}));
  EXPECT_EQ(values.size(), 6u);
}

TEST(MultiArrayFinalApiTest, FillZeroAndClearUseUniformApi) {
  using namespace jams;
  using testing::ElementsAre;

  MultiArray<int, 2> grid(2, 2);
  grid.fill(7);
  EXPECT_THAT(std::vector<int>(grid.data(), grid.data() + grid.size()),
              ElementsAre(7, 7, 7, 7));

  grid.zero();
  EXPECT_THAT(std::vector<int>(grid.data(), grid.data() + grid.size()),
              ElementsAre(0, 0, 0, 0));

  grid.clear();
  EXPECT_TRUE(grid.empty());
  EXPECT_EQ(grid.size(), 0u);
  EXPECT_EQ(grid.extent(0), 0u);
  EXPECT_EQ(grid.extent(1), 0u);
  EXPECT_EQ(grid.elements(), 0u);
  EXPECT_EQ(grid.begin(), grid.end());
}

TEST(MultiArrayFinalApiTest, FillAndZeroSupportNonNumericTriviallyCopyableValues) {
  using namespace jams;
  using Vec3 = std::array<int, 3>;

  MultiArray<Vec3, 1> values(2);
  values.fill(Vec3{1, 2, 3});
  EXPECT_EQ(values(0), (Vec3{1, 2, 3}));
  EXPECT_EQ(values(1), (Vec3{1, 2, 3}));

  values.zero();
  EXPECT_EQ(values(0), (Vec3{0, 0, 0}));
  EXPECT_EQ(values(1), (Vec3{0, 0, 0}));
}

TEST(MultiArrayFinalApiTest, ReleaseStaleStorageForwardsToSyncedMemory) {
  using namespace jams;

  if (!multiarray_cuda_device_available()) {
    GTEST_SKIP() << "CUDA runtime is enabled but no CUDA device is available";
  }

  MultiArray<int, 1> values(3);
  values(0) = 21;
  values(1) = 22;
  values(2) = 23;

  ASSERT_NE(values.device_data(), nullptr);
  values.release_stale_host();

  const int* host = values.data();
  ASSERT_NE(host, nullptr);
  EXPECT_THAT(std::vector<int>(host, host + values.size()),
              testing::ElementsAre(21, 22, 23));

  values.release_stale_device();
  EXPECT_THAT(std::vector<int>(values.data(), values.data() + values.size()),
              testing::ElementsAre(21, 22, 23));
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
