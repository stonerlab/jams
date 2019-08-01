// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_TTM_H
#define JAMS_PHYSICS_TTM_H

#include <libconfig.h++>

#include <fstream>
#include <vector>

#include "jams/core/physics.h"
#include "jams/containers/multiarray.h"

class TTMPhysics : public Physics {
 public:
  TTMPhysics(const libconfig::Setting &settings);
  ~TTMPhysics();
  void update(const int &iterations, const double &time, const double &dt);

 private:
  // calculation of pump power which is linear with input approx
  // electron temperature
  double pumpPower(double &pF) { return (1.152E20*pF); }

    jams::MultiArray<double, 1> pulseWidth;
    jams::MultiArray<double, 1> pulseFluence;
    jams::MultiArray<double, 1> pulseStartTime;
  double pumpTemp;
  double electronTemp;
  double phononTemp;
  double sinkTemp;
  std::vector<double> reversingField;

  double Ce;  // electron specific heat
  double Cl;  // phonon specific heat
  double G;   // electron coupling constant
  double Gsink;

  std::ofstream TTMFile;
};

#endif  // JAMS_PHYSICS_TTM_H
