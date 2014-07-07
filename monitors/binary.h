// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_BINARY_H
#define JAMS_MONITOR_BINARY_H

#include <fstream>
#include <vector>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"

class BinaryMonitor : public Monitor {
 public:
  BinaryMonitor(const libconfig::Setting &settings);
  ~BinaryMonitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:
    jblib::Vec3<double> slice_origin;
    jblib::Vec3<double> slice_size;
    std::vector<int>    slice_spins;
    bool is_file_overwrite_mode;
};

#endif  // JAMS_MONITOR_BINARY_H

