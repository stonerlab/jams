// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_BINARY_H
#define JAMS_MONITOR_BINARY_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "jams/core/monitor.h"
#include "jams/core/types.h"

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"

class BinaryMonitor : public Monitor {
 public:
  BinaryMonitor(const libconfig::Setting &settings);
  ~BinaryMonitor();

  void update(Solver * solver);
  bool is_converged() { return false; }
  std::string name() const {return "binary";}

 private:
    Vec3 slice_origin;
    Vec3 slice_size;
    std::vector<int>    slice_spins;
    bool is_file_overwrite_mode;
};

#endif  // JAMS_MONITOR_BINARY_H

