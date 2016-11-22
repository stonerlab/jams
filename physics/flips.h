// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_FLIPS_H
#define JAMS_PHYSICS_FLIPS_H

#include <libconfig.h++>

#include <fstream>
#include <vector>

#include "core/physics.h"

#include "jblib/containers/array.h"

class FlipsPhysics : public Physics {
 public:
  FlipsPhysics(const libconfig::Setting &settings);
  ~FlipsPhysics();
  void update(const int &iterations, const double &time, const double &dt);
 private:
  bool initialized;
};

#endif  // JAMS_PHYSICS_FLIPS_H
