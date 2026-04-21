// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_WHITE_NOISE_GENERATOR_H
#define JAMS_CUDA_WHITE_NOISE_GENERATOR_H

#if HAS_CUDA

#include <curand.h>

#include "jams/core/noise_generator.h"

class CudaWhiteNoiseGenerator : public NoiseGenerator {
 public:
  CudaWhiteNoiseGenerator(const jams::Real& temperature,
                          const jams::Real timestep,
                          int num_vectors);

  void update() override;

 private:
  jams::MultiArray<jams::Real, 1> padded_noise_;
};

#endif  // CUDA
#endif  // JAMS_CUDA_WHITE_NOISE_GENERATOR_H
