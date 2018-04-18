// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_SQUARE_H
#define JAMS_PHYSICS_SQUARE_H

#include <libconfig.h++>

#include <fstream>
#include <vector>

#include "jams/core/physics.h"

class SquarePhysics : public Physics {
 public:
  SquarePhysics(const libconfig::Setting &settings);
  ~SquarePhysics();
  void update(const int &iterations, const double &time, const double &dt);

 private:
  double PulseDuration;
  int    PulseCount;
  int    PulseTotal;
  std::vector<double> FieldStrength;
};

#endif  // JAMS_PHYSICS_SQUARE_H
