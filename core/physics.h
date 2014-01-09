// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_PHYSICS_H
#define JAMS_CORE_PHYSICS_H

#include <libconfig.h++>

enum PhysicsType{ EMPTY, FMR, MFPT, TTM, SPINWAVES, SQUARE, DYNAMICSF,
  FIELDCOOL};

class Physics {
 public:
  Physics()
  : initialised(false)
  {}

  virtual ~Physics() {}

  virtual void init(libconfig::Setting &phys);
  virtual void run(const double realtime, const double dt);
  virtual void monitor(const double realtime, const double dt);

  static Physics* Create(PhysicsType type);

 protected:
  bool initialised;
};

#endif  // JAMS_CORE_PHYSICS_H
