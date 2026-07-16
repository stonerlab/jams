// Copyright 2014 Joseph Barker. All rights reserved.

#include "jams/thermostats/cuda_quantum_spde_noise.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <jams/common.h>

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/consts.h"
#include "jams/thermostats/cuda_thermostat_quantum_spde_kernel.cuh"

namespace {

void generate_normal(curandGenerator_t generator, jams::MultiArray<jams::Real, 1>& data) {
  if (data.empty()) {
    return;
  }

#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(
      generator, data.mutable_device_data(), data.size(), 0.0, 1.0));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      generator, data.mutable_device_data(), data.size(), 0.0, 1.0));
#endif
}

void generate_normal(curandGenerator_t generator,
                     std::array<jams::MultiArray<jams::Real, 1>, 4>& data) {
  for (auto& buffer : data) {
    generate_normal(generator, buffer);
  }
}

std::size_t process_count_bytes(const int process_count) {
  return static_cast<std::size_t>(process_count) * sizeof(jams::Real);
}

}  // namespace

namespace jams {

int quantum_spde_cuda_process_count(const int spin_count) {
  if (spin_count < 0) {
    throw std::length_error("quantum-spde-gpu spin count must be non-negative");
  }

  constexpr int max_spin_count = std::numeric_limits<int>::max() / 3;
  if (spin_count > max_spin_count) {
    throw std::overflow_error(
        "quantum-spde-gpu spin-component count exceeds the int kernel limit");
  }

  return spin_count * 3;
}

std::size_t quantum_spde_cuda_curand_buffer_count(const int process_count) {
  if (process_count < 0) {
    throw std::length_error("quantum-spde-gpu process count must be non-negative");
  }

  const auto count = static_cast<std::size_t>(process_count);
  return count + (count % 2);
}

int quantum_spde_cuda_grid_size(const int process_count, const int block_size) {
  if (process_count < 0) {
    throw std::length_error("quantum-spde-gpu process count must be non-negative");
  }
  if (block_size <= 0) {
    throw std::invalid_argument("quantum-spde-gpu CUDA block size must be positive");
  }
  if (process_count == 0) {
    return 1;
  }

  const auto count = static_cast<std::size_t>(process_count);
  const auto block = static_cast<std::size_t>(block_size);
  const auto grid = (count + block - 1) / block;
  if (grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("quantum-spde-gpu CUDA grid size exceeds int");
  }
  return static_cast<int>(grid);
}

const jams::MultiArray<double, 1>& CudaQuantumSpdeNoiseGenerator::zeta0_component(
    const int component) const {
  if (component < 0 || component >= static_cast<int>(zeta0_.size())) {
    throw std::out_of_range("quantum SPDE zero-point component index out of range");
  }
  return zeta0_[static_cast<std::size_t>(component)];
}

CudaQuantumSpdeNoiseGenerator::CudaQuantumSpdeNoiseGenerator(
    const int process_count, const double delta_tau, const double omega_max,
    const bool zero_point, CudaStream& update_stream)
    : process_count_(process_count),
      curand_buffer_count_(quantum_spde_cuda_curand_buffer_count(process_count)),
      delta_tau_(delta_tau),
      omega_max_(omega_max),
      zero_point_(zero_point),
      update_stream_(update_stream) {
  const auto seed = static_cast<std::uint64_t>(jams::instance().random_generator()());
  CHECK_CURAND_STATUS(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND_STATUS(curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
  CHECK_CURAND_STATUS(curandSetStream(curand_generator_, curand_stream_.get()));
  CHECK_CURAND_STATUS(curandGenerateSeeds(curand_generator_));

  cudaEventCreateWithFlags(&curand_done_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta1a_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta1b_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta0a_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
  cudaEventCreateWithFlags(&eta0b_reusable_, cudaEventDisableTiming);
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  cudaEventRecord(eta1a_reusable_, update_stream_.get());
  cudaEventRecord(eta1b_reusable_, update_stream_.get());
  cudaEventRecord(eta0a_reusable_, update_stream_.get());
  cudaEventRecord(eta0b_reusable_, update_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  zeta5_.resize(process_count_).zero();
  zeta5p_.resize(process_count_).zero();
  zeta6_.resize(process_count_).zero();
  zeta6p_.resize(process_count_).zero();
  eta5a_.resize(curand_buffer_count_).zero();
  eta5b_.resize(curand_buffer_count_).zero();
  eta6a_.resize(curand_buffer_count_).zero();
  eta6b_.resize(curand_buffer_count_).zero();

  if (zero_point_) {
    zero_point_coefficients_ = quantum_spde_zero_point_update_coefficients(delta_tau_, omega_max_);
    for (auto& zeta : zeta0_) {
      zeta.resize(process_count_).zero();
    }
    for (auto& eta : eta0a_) {
      eta.resize(curand_buffer_count_).zero();
    }
    for (auto& eta : eta0b_) {
      eta.resize(curand_buffer_count_).zero();
    }
  }

  generate_random_buffers();
}

CudaQuantumSpdeNoiseGenerator::~CudaQuantumSpdeNoiseGenerator() {
  synchronize();

  if (eta0b_reusable_ != nullptr) {
    cudaEventDestroy(eta0b_reusable_);
    eta0b_reusable_ = nullptr;
  }

  if (eta0a_reusable_ != nullptr) {
    cudaEventDestroy(eta0a_reusable_);
    eta0a_reusable_ = nullptr;
  }

  if (eta1b_reusable_ != nullptr) {
    cudaEventDestroy(eta1b_reusable_);
    eta1b_reusable_ = nullptr;
  }

  if (eta1a_reusable_ != nullptr) {
    cudaEventDestroy(eta1a_reusable_);
    eta1a_reusable_ = nullptr;
  }

  if (curand_done_ != nullptr) {
    cudaEventDestroy(curand_done_);
    curand_done_ = nullptr;
  }

  if (curand_generator_ != nullptr) {
    curandDestroyGenerator(curand_generator_);
    curand_generator_ = nullptr;
  }
}

void CudaQuantumSpdeNoiseGenerator::generate_random_buffers() {
  if (zero_point_) {
    generate_normal(curand_generator_, eta0a_);
    generate_normal(curand_generator_, eta0b_);
  }
  generate_normal(curand_generator_, eta5a_);
  generate_normal(curand_generator_, eta5b_);
  generate_normal(curand_generator_, eta6a_);
  generate_normal(curand_generator_, eta6b_);

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CudaQuantumSpdeNoiseGenerator::prepare_fixed_temperature_coefficients(
    const jams::Real temperature) {
  const double reduced_delta_tau = delta_tau_ * static_cast<double>(temperature);
  if (reduced_delta_tau <= 0.0) {
    fast_coefficients_valid_ = false;
    fast_temperature_ = temperature;
    return;
  }

  fast_factor5_ = quantum_spde_bose_update_coefficients(
      kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
  fast_factor6_ = quantum_spde_bose_update_coefficients(
      kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);
  fast_temperature_ = temperature;
  fast_coefficients_valid_ = true;
}

void CudaQuantumSpdeNoiseGenerator::prepare_temperature_profile_coefficients(
    const jams::MultiArray<jams::Real, 1>& temperature) {
  if (temperature.size() != static_cast<std::size_t>(process_count_)) {
    throw std::runtime_error("quantum SPDE temperature profile size does not match process count");
  }

  profile_factor5_.resize(process_count_);
  profile_factor6_.resize(process_count_);
  profile_stationary_factor5_.resize(process_count_);
  profile_stationary_factor6_.resize(process_count_);
  profile_has_positive_temperature_ = false;

  const auto* temperature_host = temperature.host_data();
  auto factor5 = profile_factor5_.mutable_host_view();
  auto factor6 = profile_factor6_.mutable_host_view();
  auto stationary5 = profile_stationary_factor5_.mutable_host_view();
  auto stationary6 = profile_stationary_factor6_.mutable_host_view();

  for (auto x = 0; x < process_count_; ++x) {
    const double reduced_delta_tau =
        delta_tau_ * static_cast<double>(temperature_host[x]);
    if (reduced_delta_tau <= 0.0) {
      factor5(x) = {};
      factor6(x) = {};
      stationary5(x) = {};
      stationary6(x) = {};
      continue;
    }

    profile_has_positive_temperature_ = true;
    factor5(x) = quantum_spde_bose_update_coefficients(
        kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
    factor6(x) = quantum_spde_bose_update_coefficients(
        kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);
    stationary5(x) = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
    stationary6(x) = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);
  }
}

void CudaQuantumSpdeNoiseGenerator::zero_state() {
  synchronize();

  zeta5_.zero();
  zeta5p_.zero();
  zeta6_.zero();
  zeta6p_.zero();
  if (zero_point_) {
    for (auto& zeta : zeta0_) {
      zeta.zero();
    }
  }
}

void CudaQuantumSpdeNoiseGenerator::initialize(const Initialization initialization,
                                              const jams::Real temperature) {
  if (initialization == Initialization::Stationary) {
    initialize_stationary(temperature);
    return;
  }

  zero_state();
  generate_random_buffers();
  prepare_fixed_temperature_coefficients(temperature);
}

void CudaQuantumSpdeNoiseGenerator::initialize(
    const Initialization initialization,
    const jams::MultiArray<jams::Real, 1>& temperature) {
  prepare_temperature_profile_coefficients(temperature);
  if (initialization == Initialization::Stationary) {
    initialize_stationary(temperature);
    return;
  }

  zero_state();
  generate_random_buffers();
}

void CudaQuantumSpdeNoiseGenerator::initialize_stationary(const jams::Real temperature) {
  zero_state();

  const int block_size = 128;
  const int grid_size = quantum_spde_cuda_grid_size(process_count_, block_size);
  bool consumed_random_numbers = false;

  cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);

  const double reduced_delta_tau = delta_tau_ * temperature;
  if (reduced_delta_tau > 0.0) {
    const auto factor5 = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma5, kQuantumSpdeOmega5, reduced_delta_tau);
    const auto factor6 = quantum_spde_stationary_bose_cholesky(
        kQuantumSpdeGamma6, kQuantumSpdeOmega6, reduced_delta_tau);
    cuda_thermostat_quantum_spde_stationary_no_zero_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
        zeta6p_.mutable_device_data(), eta5a_.device_data(), eta5b_.device_data(),
        eta6a_.device_data(), eta6b_.device_data(),
        factor5.l00, factor5.l10, factor5.l11, factor6.l00, factor6.l10, factor6.l11,
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (zero_point_) {
    const double zero_point_delta_tau = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
    cuda_thermostat_quantum_spde_stationary_zero_point_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        zeta0_[0].mutable_device_data(), zeta0_[1].mutable_device_data(),
        zeta0_[2].mutable_device_data(), zeta0_[3].mutable_device_data(),
        eta0a_[0].device_data(), eta0a_[1].device_data(),
        eta0a_[2].device_data(), eta0a_[3].device_data(),
        static_cast<jams::Real>(zero_point_delta_tau),
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (consumed_random_numbers) {
    update_stream_.synchronize();
    generate_random_buffers();
  }

  prepare_fixed_temperature_coefficients(temperature);
}

void CudaQuantumSpdeNoiseGenerator::initialize_stationary(
    const jams::MultiArray<jams::Real, 1>& temperature) {
  prepare_temperature_profile_coefficients(temperature);
  zero_state();

  const int block_size = 128;
  const int grid_size = quantum_spde_cuda_grid_size(process_count_, block_size);
  bool consumed_random_numbers = false;

  cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);

  if (profile_has_positive_temperature_) {
    cuda_thermostat_quantum_spde_stationary_no_zero_profile_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
        zeta6p_.mutable_device_data(), eta5a_.device_data(), eta5b_.device_data(),
        eta6a_.device_data(), eta6b_.device_data(),
        profile_stationary_factor5_.device_data(), profile_stationary_factor6_.device_data(),
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (zero_point_) {
    const double zero_point_delta_tau = (kHBarIU * omega_max_ * delta_tau_) / kBoltzmannIU;
    cuda_thermostat_quantum_spde_stationary_zero_point_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        zeta0_[0].mutable_device_data(), zeta0_[1].mutable_device_data(),
        zeta0_[2].mutable_device_data(), zeta0_[3].mutable_device_data(),
        eta0a_[0].device_data(), eta0a_[1].device_data(),
        eta0a_[2].device_data(), eta0a_[3].device_data(),
        static_cast<jams::Real>(zero_point_delta_tau),
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    consumed_random_numbers = true;
  }

  if (consumed_random_numbers) {
    update_stream_.synchronize();
    generate_random_buffers();
  }
}

void CudaQuantumSpdeNoiseGenerator::warmup(const unsigned steps,
                                          const jams::Real temperature,
                                          jams::Real* noise,
                                          const jams::Real* sigma) {
  for (auto i = 0u; i < steps; ++i) {
    update(noise, sigma, temperature);
  }
}

void CudaQuantumSpdeNoiseGenerator::warmup(
    const unsigned steps,
    const jams::MultiArray<jams::Real, 1>& temperature,
    jams::Real* noise,
    const jams::Real* sigma) {
  for (auto i = 0u; i < steps; ++i) {
    update(noise, sigma, temperature);
  }
}

void CudaQuantumSpdeNoiseGenerator::update(jams::Real* noise,
                                           const jams::Real* sigma,
                                           const jams::Real temperature) {
  const int block_size = 128;
  const int grid_size = quantum_spde_cuda_grid_size(process_count_, block_size);

  if (temperature == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise, 0, process_count_bytes(process_count_), update_stream_.get()));

    if (zero_point_) {
      std::swap(eta0a_, eta0b_);
      std::swap(eta0a_reusable_, eta0b_reusable_);

      cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
      cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);
      cuda_thermostat_quantum_spde_zero_point_fast_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
          noise,
          zeta0_[0].mutable_device_data(), zeta0_[1].mutable_device_data(),
          zeta0_[2].mutable_device_data(), zeta0_[3].mutable_device_data(),
          eta0b_[0].device_data(), eta0b_[1].device_data(),
          eta0b_[2].device_data(), eta0b_[3].device_data(),
          sigma, zero_point_coefficients_,
          process_count_);
      DEBUG_CHECK_CUDA_ASYNC_STATUS;
      cudaEventRecord(eta0b_reusable_, update_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS

      generate_normal(curand_generator_, eta0a_);

      cudaEventRecord(curand_done_, curand_stream_.get());
      DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

    return;
  }

  const double reduced_delta_tau = delta_tau_ * temperature;
  const bool use_fast_no_zero = fast_coefficients_valid_ && temperature == fast_temperature_;

  // The refill target was used by an earlier update before the swap. Each
  // buffer pair carries its own reusable event, so CURAND only waits for the
  // pair it is about to overwrite.
  swap(eta5a_, eta5b_);
  swap(eta6a_, eta6b_);
  std::swap(eta1a_reusable_, eta1b_reusable_);

  cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
  cudaStreamWaitEvent(curand_stream_.get(), eta1a_reusable_, 0);
  if (use_fast_no_zero) {
    cuda_thermostat_quantum_spde_no_zero_fast_kernel<<<grid_size, block_size, 0, update_stream_.get() >>> (
      noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
      zeta6p_.mutable_device_data(), eta5b_.device_data(), eta6b_.device_data(),
      sigma, fast_factor5_, fast_factor6_,
      temperature, process_count_);
  } else {
    cuda_thermostat_quantum_spde_no_zero_kernel<<<grid_size, block_size, 0, update_stream_.get() >>> (
      noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
      zeta6p_.mutable_device_data(), eta5b_.device_data(), eta6b_.device_data(),
      sigma, reduced_delta_tau,
      temperature, process_count_);
    prepare_fixed_temperature_coefficients(temperature);
  }
  DEBUG_CHECK_CUDA_ASYNC_STATUS;
  cudaEventRecord(eta1b_reusable_, update_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS

  generate_normal(curand_generator_, eta5a_);
  generate_normal(curand_generator_, eta6a_);

  if (zero_point_) {
    std::swap(eta0a_, eta0b_);
    std::swap(eta0a_reusable_, eta0b_reusable_);
    cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);

    cuda_thermostat_quantum_spde_zero_point_fast_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        noise,
        zeta0_[0].mutable_device_data(), zeta0_[1].mutable_device_data(),
        zeta0_[2].mutable_device_data(), zeta0_[3].mutable_device_data(),
        eta0b_[0].device_data(), eta0b_[1].device_data(),
        eta0b_[2].device_data(), eta0b_[3].device_data(),
        sigma, zero_point_coefficients_,
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    cudaEventRecord(eta0b_reusable_, update_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS

    generate_normal(curand_generator_, eta0a_);
  }

  cudaEventRecord(curand_done_, curand_stream_.get());
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CudaQuantumSpdeNoiseGenerator::update(
    jams::Real* noise,
    const jams::Real* sigma,
    const jams::MultiArray<jams::Real, 1>& temperature) {
  const int block_size = 128;
  const int grid_size = quantum_spde_cuda_grid_size(process_count_, block_size);

  if (!profile_has_positive_temperature_) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(noise, 0, process_count_bytes(process_count_), update_stream_.get()));
  } else {
    swap(eta5a_, eta5b_);
    swap(eta6a_, eta6b_);
    std::swap(eta1a_reusable_, eta1b_reusable_);

    cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
    cudaStreamWaitEvent(curand_stream_.get(), eta1a_reusable_, 0);
    cuda_thermostat_quantum_spde_no_zero_profile_fast_kernel<<<grid_size, block_size, 0, update_stream_.get() >>> (
      noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(), zeta6_.mutable_device_data(),
      zeta6p_.mutable_device_data(), eta5b_.device_data(), eta6b_.device_data(),
      sigma, profile_factor5_.device_data(),
      profile_factor6_.device_data(), temperature.device_data(), process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    cudaEventRecord(eta1b_reusable_, update_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS

    generate_normal(curand_generator_, eta5a_);
    generate_normal(curand_generator_, eta6a_);
  }

  if (zero_point_) {
    std::swap(eta0a_, eta0b_);
    std::swap(eta0a_reusable_, eta0b_reusable_);
    cudaStreamWaitEvent(update_stream_.get(), curand_done_, 0);
    cudaStreamWaitEvent(curand_stream_.get(), eta0a_reusable_, 0);

    cuda_thermostat_quantum_spde_zero_point_fast_kernel <<< grid_size, block_size, 0, update_stream_.get() >>> (
        noise,
        zeta0_[0].mutable_device_data(), zeta0_[1].mutable_device_data(),
        zeta0_[2].mutable_device_data(), zeta0_[3].mutable_device_data(),
        eta0b_[0].device_data(), eta0b_[1].device_data(),
        eta0b_[2].device_data(), eta0b_[3].device_data(),
        sigma, zero_point_coefficients_,
        process_count_);
    DEBUG_CHECK_CUDA_ASYNC_STATUS;
    cudaEventRecord(eta0b_reusable_, update_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS

    generate_normal(curand_generator_, eta0a_);
  }

  if (profile_has_positive_temperature_ || zero_point_) {
    cudaEventRecord(curand_done_, curand_stream_.get());
    DEBUG_CHECK_CUDA_ASYNC_STATUS
  }
}

void CudaQuantumSpdeNoiseGenerator::update_with_gaussian_replay(
    jams::Real* noise, const jams::Real* sigma, const jams::Real temperature,
    const jams::Real* eta5, const jams::Real* eta6) {
  if (temperature <= jams::Real{0.0}) {
    throw std::invalid_argument(
        "quantum SPDE Gaussian replay requires a positive temperature");
  }
  if (zero_point_) {
    throw std::invalid_argument(
        "quantum SPDE Gaussian replay does not include zero-point noise");
  }
  if (noise == nullptr || sigma == nullptr || eta5 == nullptr || eta6 == nullptr) {
    throw std::invalid_argument(
        "quantum SPDE Gaussian replay requires non-null device arrays");
  }

  prepare_fixed_temperature_coefficients(temperature);
  const int block_size = 128;
  const int grid_size = quantum_spde_cuda_grid_size(process_count_, block_size);
  cuda_thermostat_quantum_spde_no_zero_fast_kernel<<<
      grid_size, block_size, 0, update_stream_.get()>>>(
      noise, zeta5_.mutable_device_data(), zeta5p_.mutable_device_data(),
      zeta6_.mutable_device_data(), zeta6p_.mutable_device_data(), eta5, eta6,
      sigma, fast_factor5_, fast_factor6_, temperature, process_count_);
  DEBUG_CHECK_CUDA_ASYNC_STATUS
}

void CudaQuantumSpdeNoiseGenerator::synchronize() {
  update_stream_.synchronize();
  curand_stream_.synchronize();
}

}  // namespace jams
