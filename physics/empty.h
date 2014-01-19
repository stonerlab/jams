// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_EMPTY_H
#define JAMS_PHYSICS_EMPTY_H

#include <libconfig.h++>

#include "core/physics.h"

class EmptyPhysics : public Physics {
 public:
  EmptyPhysics() : initialized(true) {}
  ~EmptyPhysics();
  void initialize(libconfig::Setting &phys);
  void run(double realtime, const double dt);
  virtual void monitor(double realtime, const double dt);
 private:
  bool initialized;
};

#endif  // JAMS_PHYSICS_EMPTY_H
