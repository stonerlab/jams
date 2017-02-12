// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_FIELDCOOL_H
#define JAMS_PHYSICS_FIELDCOOL_H

#include <libconfig.h++>

#include <vector>

#include "jams/core/physics.h"

class FieldCoolPhysics : public Physics {
 public:
  FieldCoolPhysics(const libconfig::Setting &settings);
  ~FieldCoolPhysics() {};
  void update(const int &iterations, const double &time, const double &dt);

 private:
  std::vector<double> initField;
  std::vector<double> finalField;
  std::vector<int>    deltaH;
  double initTemp;
  double finalTemp;
  double coolTime;
  int    TSteps;
  double deltaT;
  double t_step;
  double t_eq;
  double integration_time_step_;
  bool stepToggle;
  bool initialized;
};

#endif  // JAMS_PHYSICS_FIELDCOOL_H
