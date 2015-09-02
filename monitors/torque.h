// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_TORQUE_H
#define JAMS_MONITOR_TORQUE_H

#include <fstream>

#include "core/monitor.h"
#include "core/stats.h"

#include "jblib/containers/vec.h"

class TorqueMonitor : public Monitor {
 public:
  TorqueMonitor(const libconfig::Setting &settings);
  ~TorqueMonitor();

  void update(Solver * solver);

 private:
  jblib::Vec3<double> torque_;
  std::ofstream outfile;
  Stats torque_stats_;
};

#endif  // JAMS_MONITOR_TORQUE_H

