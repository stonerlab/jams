// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CPU_WHITE_NOISE_GENERATOR_H
#define JAMS_CPU_WHITE_NOISE_GENERATOR_H

#include "jams/core/noise_generator.h"

class CpuWhiteNoiseGenerator : public NoiseGenerator {
 public:
  CpuWhiteNoiseGenerator(const jams::Real& temperature,
                         const jams::Real timestep,
                         int num_vectors);

  void update() override;
};

#endif  // JAMS_CPU_WHITE_NOISE_GENERATOR_H
