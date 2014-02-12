// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_FIELDCOOL_H
#define JAMS_PHYSICS_FIELDCOOL_H

#include <libconfig.h++>

#include <fstream>
#include <vector>

#include "core/physics.h"

class FieldCoolPhysics : public Physics {
 public:
  FieldCoolPhysics()
  : initField(3, 0.0),
  finalField(3, 0.0),
  deltaH(3, 0.0),
  initTemp(0.0),
  finalTemp(0.0),
  coolTime(0.0),
  TSteps(0),
  deltaT(0),
  t_step(0),
  t_eq(0),
  stepToggle(false),
  initialized(false)
  {}
  ~FieldCoolPhysics();
  void initialize(libconfig::Setting &phys);
  void run(double realtime, const double dt);
  virtual void monitor(double realtime, const double dt);

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
  bool stepToggle;
  bool initialized;
};

#endif  // JAMS_PHYSICS_FIELDCOOL_H
