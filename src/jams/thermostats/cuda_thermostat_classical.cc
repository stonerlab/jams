// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <iostream>

#include <jams/common.h>

#include "jams/cuda/cuda_common.h"
#include "jams/helpers/error.h"
#include "jams/thermostats/cuda_thermostat_classical.h"

CudaWhiteNoiseGenerator::CudaWhiteNoiseGenerator(const jams::Real& temperature,
                                                 const jams::Real timestep,
                                                 const int num_vectors)
    : NoiseGenerator(temperature, num_vectors),
      padded_noise_(noise_.elements() + (noise_.elements() % 2)) {
  zero(padded_noise_);
  std::cout << "\n  initialising classical-gpu noise generator\n";
}

void CudaWhiteNoiseGenerator::update() {
  if (this->temperature() == 0) {
    CHECK_CUDA_STATUS(cudaMemsetAsync(
        noise_.device_data(), 0, noise_.bytes(), cuda_stream_.get()));
    return;
  }

  CHECK_CURAND_STATUS(
      curandSetStream(jams::instance().curand_generator(), cuda_stream_.get()));

#ifdef DO_MIXED_PRECISION
  CHECK_CURAND_STATUS(curandGenerateNormal(
      jams::instance().curand_generator(),
      padded_noise_.device_data(),
      padded_noise_.elements(),
      0.0f,
      std::sqrt(static_cast<float>(this->temperature()))));
#else
  CHECK_CURAND_STATUS(curandGenerateNormalDouble(
      jams::instance().curand_generator(),
      padded_noise_.device_data(),
      padded_noise_.elements(),
      0.0,
      std::sqrt(static_cast<double>(this->temperature()))));
#endif

  CHECK_CUDA_STATUS(cudaMemcpyAsync(
      noise_.device_data(),
      padded_noise_.device_data(),
      noise_.bytes(),
      cudaMemcpyDeviceToDevice,
      cuda_stream_.get()));
}
