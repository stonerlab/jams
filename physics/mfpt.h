// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_PHYSICS_MFPT_H
#define JAMS_PHYSICS_MFPT_H

#include <libconfig.h++>

#include <fstream>
#include <vector>

#include "core/physics.h"

class MFPTPhysics : public Physics {
 public:
  MFPTPhysics()
  : initialized(false),
  maskArray(),
  MFPTFile()
  {}
  ~MFPTPhysics();
  void initialize(libconfig::Setting &phys);
  void run(double realtime, const double dt);
  virtual void monitor(double realtime, const double dt);

 private:
  bool initialized;
  std::vector<double> maskArray;
  std::ofstream MFPTFile;
};

#endif  // JAMS_PHYSICS_MFPT_H
