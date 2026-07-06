#include "gtest/gtest.h"

#include "jams/solvers/cuda_solver_functions.cuh"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <limits>

namespace {

bool cuda_device_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

template <typename GyroParam, typename AlphaParam>
__global__ void cuda_solver_parameter_test_kernel(
    const GyroParam gyro,
    const AlphaParam alpha,
    jams::Real* out,
    const unsigned count) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  out[2u * idx + 0u] = gyro.get(idx);
  out[2u * idx + 1u] = alpha.get(idx);
}

template <typename FieldScaleParam>
__global__ void cuda_field_scale_test_kernel(
    const FieldScaleParam field_scale,
    const jams::Real* field,
    jams::Real* out,
    const unsigned count) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  out[idx] = field_scale.scale(field[idx], idx);
}

void run_wrapper_case(
    const CudaSpinParameterChoice gyro_choice,
    const CudaSpinParameterChoice alpha_choice,
    const std::array<jams::Real, 4>& expected_gyro,
    const std::array<jams::Real, 4>& expected_alpha) {
  constexpr unsigned kCount = 4;

  jams::MultiArray<jams::Real, 1> gyro(kCount);
  jams::MultiArray<jams::Real, 1> alpha(kCount);
  jams::MultiArray<jams::Real, 1> out(2u * kCount);

  auto gyro_values = gyro.mutable_host_span();
  auto alpha_values = alpha.mutable_host_span();
  for (unsigned i = 0; i < kCount; ++i) {
    gyro_values[i] = static_cast<jams::Real>(10.0 + i);
    alpha_values[i] = static_cast<jams::Real>(20.0 + i);
  }

  dispatch_cuda_spin_parameters(
      gyro_choice,
      alpha_choice,
      gyro.device_data(),
      alpha.device_data(),
      [&](const auto gyro_param, const auto alpha_param) {
        cuda_solver_parameter_test_kernel<<<1, 32>>>(
            gyro_param,
            alpha_param,
            out.mutable_device_data(),
            kCount);
      });

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto out_values = out.host_span();
  for (unsigned i = 0; i < kCount; ++i) {
    EXPECT_EQ(out_values[2u * i + 0u], expected_gyro[i]);
    EXPECT_EQ(out_values[2u * i + 1u], expected_alpha[i]);
  }
}

void run_field_scale_case(
    const CudaFieldScaleChoice choice,
    const std::array<jams::Real, 4>& expected) {
  constexpr unsigned kCount = 4;

  jams::MultiArray<jams::Real, 1> inv_mus(kCount);
  jams::MultiArray<jams::Real, 1> field(kCount);
  jams::MultiArray<jams::Real, 1> out(kCount);

  auto inv_mus_values = inv_mus.mutable_host_span();
  auto field_values = field.mutable_host_span();
  inv_mus_values[0] = jams::Real{0.5};
  inv_mus_values[1] = jams::Real{1.0 / 3.0};
  inv_mus_values[2] = jams::Real{0.0};
  inv_mus_values[3] = jams::Real{0.2};
  for (unsigned i = 0; i < kCount; ++i) {
    field_values[i] = static_cast<jams::Real>(20.0 + i);
  }

  dispatch_cuda_field_scale(
      choice,
      inv_mus,
      [&](const auto field_scale) {
        cuda_field_scale_test_kernel<<<1, 32>>>(
            field_scale,
            field.device_data(),
            out.mutable_device_data(),
            kCount);
      });

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto out_values = out.host_span();
  for (unsigned i = 0; i < kCount; ++i) {
    EXPECT_EQ(out_values[i], expected[i]);
  }
}

}  // namespace

TEST(CudaSolverParameterChoiceTest, EmptyArrayIsNotUniform) {
  const jams::MultiArray<jams::Real, 1> values;

  const auto choice = cuda_spin_parameter_choice(values);

  EXPECT_FALSE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_value, jams::Real{0.0});
}

TEST(CudaSolverParameterChoiceTest, SingleElementIsUniform) {
  jams::MultiArray<jams::Real, 1> values(1);
  values(0) = jams::Real{1.25};

  const auto choice = cuda_spin_parameter_choice(values);

  EXPECT_TRUE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_value, jams::Real{1.25});
}

TEST(CudaSolverParameterChoiceTest, EqualElementsAreUniform) {
  jams::MultiArray<jams::Real, 1> values(4);
  values.fill(jams::Real{0.5});

  const auto choice = cuda_spin_parameter_choice(values);

  EXPECT_TRUE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_value, jams::Real{0.5});
}

TEST(CudaSolverParameterChoiceTest, DifferingElementIsNotUniform) {
  jams::MultiArray<jams::Real, 1> values(4);
  values.fill(jams::Real{0.5});
  values(2) = jams::Real{0.75};

  const auto choice = cuda_spin_parameter_choice(values);

  EXPECT_FALSE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_value, jams::Real{0.5});
}

TEST(CudaSolverParameterChoiceTest, NearButNotExactElementsAreNotUniform) {
  jams::MultiArray<jams::Real, 1> values(2);
  values(0) = jams::Real{0.5};
  values(1) = std::nextafter(
      jams::Real{0.5},
      std::numeric_limits<jams::Real>::max());

  const auto choice = cuda_spin_parameter_choice(values);

  EXPECT_FALSE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_value, jams::Real{0.5});
}

TEST(CudaFieldScaleChoiceTest, EmptyArrayIsNotUniform) {
  const jams::MultiArray<jams::Real, 1> values;

  const auto choice = cuda_field_scale_choice(values);

  EXPECT_FALSE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_inv_mus, jams::Real{0.0});
}

TEST(CudaFieldScaleChoiceTest, UniformArrayStoresInverseMoment) {
  jams::MultiArray<jams::Real, 1> values(4);
  values.fill(jams::Real{0.5});

  const auto choice = cuda_field_scale_choice(values);

  EXPECT_TRUE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_inv_mus, jams::Real{0.5});
}

TEST(CudaFieldScaleChoiceTest, DifferingElementIsNotUniform) {
  jams::MultiArray<jams::Real, 1> values(4);
  values.fill(jams::Real{0.5});
  values(2) = jams::Real{0.25};

  const auto choice = cuda_field_scale_choice(values);

  EXPECT_FALSE(choice.is_uniform);
  EXPECT_EQ(choice.uniform_inv_mus, jams::Real{0.0});
}

TEST(CudaSolverParameterWrapperTest, ReturnsExpectedValuesForAllWrapperCombinations) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  constexpr std::array<jams::Real, 4> kPerSpinGyro = {
      jams::Real{10.0}, jams::Real{11.0}, jams::Real{12.0}, jams::Real{13.0}};
  constexpr std::array<jams::Real, 4> kPerSpinAlpha = {
      jams::Real{20.0}, jams::Real{21.0}, jams::Real{22.0}, jams::Real{23.0}};
  constexpr std::array<jams::Real, 4> kUniformGyro = {
      jams::Real{3.0}, jams::Real{3.0}, jams::Real{3.0}, jams::Real{3.0}};
  constexpr std::array<jams::Real, 4> kUniformAlpha = {
      jams::Real{0.1}, jams::Real{0.1}, jams::Real{0.1}, jams::Real{0.1}};

  run_wrapper_case(
      {false, jams::Real{0.0}},
      {false, jams::Real{0.0}},
      kPerSpinGyro,
      kPerSpinAlpha);

  run_wrapper_case(
      {true, jams::Real{3.0}},
      {true, jams::Real{0.1}},
      kUniformGyro,
      kUniformAlpha);

  run_wrapper_case(
      {true, jams::Real{3.0}},
      {false, jams::Real{0.0}},
      kUniformGyro,
      kPerSpinAlpha);

  run_wrapper_case(
      {false, jams::Real{0.0}},
      {true, jams::Real{0.1}},
      kPerSpinGyro,
      kUniformAlpha);
}

TEST(CudaFieldScaleWrapperTest, ScalesFieldsForUniformAndPerSpinMoment) {
  if (!cuda_device_available()) {
    GTEST_SKIP() << "CUDA device not available";
  }

  constexpr std::array<jams::Real, 4> kUniformExpected = {
      jams::Real{10.0}, jams::Real{10.5}, jams::Real{11.0}, jams::Real{11.5}};
  constexpr std::array<jams::Real, 4> kPerSpinExpected = {
      jams::Real{10.0},
      jams::Real{21.0 / 3.0},
      jams::Real{0.0},
      jams::Real{4.6}};

  run_field_scale_case(
      {true, jams::Real{0.5}},
      kUniformExpected);

  run_field_scale_case(
      {false, jams::Real{0.0}},
      kPerSpinExpected);
}
