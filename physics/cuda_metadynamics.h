// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_METADYNAMICS_H
#define JAMS_PHYSICS_METADYNAMICS_H

#include <libconfig.h++>

#include "core/physics.h"

class CudaMetadynamicsPhysics : public Physics {
 public:
  CudaMetadynamicsPhysics(const libconfig::Setting &settings);
  ~CudaMetadynamicsPhysics();
  void update(const int &iterations, const double &time, const double &dt);

 private:
};

#endif  // JAMS_PHYSICS_METADYNAMICS_H
