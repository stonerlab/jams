// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_PING_H
#define JAMS_PHYSICS_PING_H

#include <libconfig.h++>

#include "jams/core/physics.h"

class PingPhysics : public Physics {
 public:
  PingPhysics(const libconfig::Setting &settings);
  ~PingPhysics();
  void update(const int &iterations, const double &time, const double &dt);
 private:
  bool initialized;
};

#endif  // JAMS_PHYSICS_PING_H
