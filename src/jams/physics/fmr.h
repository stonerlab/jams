// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_FMR_H
#define JAMS_PHYSICS_FMR_H

#include <libconfig.h++>

#include <fstream>
#include <vector>
#include <jams/containers/multiarray.h>

#include "jams/core/physics.h"

class FMRPhysics : public Physics {
 public:
  FMRPhysics(const libconfig::Setting &settings);
  ~FMRPhysics();
  void update(const int &iterations, const double &time, const double &dt);
 private:
  bool initialized;
  double ACFieldFrequency;
  std::vector<double> ACFieldStrength;
  std::vector<double> DCFieldStrength;
  std::ofstream PSDFile;
  jams::MultiArray<double, 1> PSDIntegral;
};

#endif  // JAMS_PHYSICS_FMR_H
