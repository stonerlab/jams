// Copyright 2014 Joseph Barker. All rights reserved.

#include "flips.h"

#include <libconfig.h++>

#include "jams/helpers/random.h"
#include "jams/core/globals.h"
#include "jblib/containers/array.h"

FlipsPhysics::FlipsPhysics(const libconfig::Setting &settings)
: Physics(settings) {
  std::uniform_real_distribution<> uniform_distribution;

  int count = 0;
  for (auto i = 0; i < globals::num_spins; ++i) {
    if (uniform_distribution(random_generator_) < 0.1) {
      for (auto j = 0; j < 3; ++j) {
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
