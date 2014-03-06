// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_MFPT_H
#define JAMS_PHYSICS_MFPT_H

#include <libconfig.h++>

#include <fstream>
#include <vector>

#include "core/physics.h"

class MFPTPhysics : public Physics {
 public:
  MFPTPhysics(const libconfig::Setting &settings);
  ~MFPTPhysics();

  void update(const int &iterations, const double &time, const double &dt);

 private:
  std::vector<double> maskArray;
  std::ofstream MFPTFile;
};

#endif  // JAMS_PHYSICS_MFPT_H
