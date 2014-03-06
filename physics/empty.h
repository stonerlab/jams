// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_EMPTY_H
#define JAMS_PHYSICS_EMPTY_H

#include <libconfig.h++>

#include "core/physics.h"

class EmptyPhysics : public Physics {
 public:
  EmptyPhysics(const libconfig::Setting &settings) : Physics(settings) {}
  ~EmptyPhysics() {};
  void update(const int &iterations, const double &time, const double &dt) {};
 private:
};

#endif  // JAMS_PHYSICS_EMPTY_H