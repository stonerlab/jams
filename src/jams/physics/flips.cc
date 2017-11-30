// Copyright 2014 Joseph Barker. All rights reserved.

#include "flips.h"

#include <libconfig.h++>

#include "jams/core/rand.h"
#include "jams/core/globals.h"
#include "jblib/containers/array.h"

FlipsPhysics::FlipsPhysics(const libconfig::Setting &settings)
: Physics(settings) {

  int count = 0;
  for (int i = 0; i < globals::num_spins; ++i) {

    if (rng->uniform() < 0.1) {
      for (int j = 0; j < 3; ++j) {
        globals::s(i, j) = -globals::s(i, j);
      }
      count++;
    }
  }
  std::cout << "  spins flipped " << count << "\n";

  initialized = true;
}

FlipsPhysics::~FlipsPhysics() {
}

void FlipsPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;
}
