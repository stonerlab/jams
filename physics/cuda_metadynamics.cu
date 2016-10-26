// Copyright 2014 Joseph Barker. All rights reserved.

#include "physics/cuda_metadynamics.h"

#include <libconfig.h++>

#include "core/globals.h"

CudaMetadynamicsPhysics::CudaMetadynamicsPhysics(const libconfig::Setting &settings)
  : Physics(settings) {
  using namespace globals;

  output.write("  * CUDA metadynamics physics module\n");
}

CudaMetadynamicsPhysics::~CudaMetadynamicsPhysics() {
}

void CudaMetadynamicsPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;

  // calculate collective variables

  // if t == t_gaussian store new gaussian

  // compute derivatives

  // compute fields

}
