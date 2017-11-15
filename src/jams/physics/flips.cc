// Copyright 2014 Joseph Barker. All rights reserved.

#include "flips.h"

#include <libconfig.h++>

#include "jams/core/output.h"
#include "jams/core/rand.h"
#include "jams/core/globals.h"
#include "jblib/containers/array.h"

FlipsPhysics::FlipsPhysics(const libconfig::Setting &settings)
: Physics(settings) {

  ::output->write("\nInitialising Flips Physics module...\n");

  int count = 0;
  for (int i = 0; i < globals::num_spins; ++i) {

    if (rng->uniform() < 0.1) {
      for (int j = 0; j < 3; ++j) {
        globals::s(i, j) = -globals::s(i, j);
      }
      count++;
    }
  }
  ::output->write("  %d spins flipped\n", count);


  initialized = true;
}

FlipsPhysics::~FlipsPhysics() {
}

void FlipsPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;
}
