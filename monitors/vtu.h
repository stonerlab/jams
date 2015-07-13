// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_VTU_H
#define JAMS_MONITOR_VTU_H

#include <fstream>
#include <vector>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class VtuMonitor : public Monitor {
 public:
  VtuMonitor(const libconfig::Setting &settings);
  ~VtuMonitor();

  void update(const Solver * const solver);

 private:
    int num_slice_points;
    jblib::Vec3<double> slice_origin;
    jblib::Vec3<double> slice_size;
    std::vector<int>        slice_spins;
    jblib::Array<int, 1>    types_binary_data;
    jblib::Array<float, 2>  points_binary_data;
    jblib::Array<double, 2> spins_binary_data;


};

#endif  // JAMS_MONITOR_VTU_H

