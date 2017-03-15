// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_XYZ_H
#define JAMS_MONITOR_XYZ_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"

class Solver;

class XyzMonitor : public Monitor {
 public:
  XyzMonitor(const libconfig::Setting &settings);
  ~XyzMonitor();

  void update(Solver * solver);
  bool is_converged() { return false; }
  std::string name() const {return "xyz";}


 private:
    jblib::Vec3<double> slice_origin;
    jblib::Vec3<double> slice_size;
    std::vector<int>    slice_spins;
};

#endif  // JAMS_MONITOR_XYZ_H

