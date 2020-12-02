// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_FLIPS_H
#define JAMS_PHYSICS_FLIPS_H

#include <libconfig.h++>
#include <pcg_random.hpp>
#include <random>
#include "jams/common.h"
#include "jams/helpers/random.h"

#include "jams/core/physics.h"

class FlipsPhysics : public Physics {
 public:
  FlipsPhysics(const libconfig::Setting &settings);
  ~FlipsPhysics();
  void update(const int &iterations, const double &time, const double &dt);
 private:
  bool initialized;
    pcg32_k1024 random_generator_ = pcg_extras::seed_seq_from<pcg32>(jams::instance().random_generator()());

};

#endif  // JAMS_PHYSICS_FLIPS_H
