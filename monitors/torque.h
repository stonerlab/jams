// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_TORQUE_H
#define JAMS_MONITOR_TORQUE_H

#include <fstream>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/vec.h"

class TorqueMonitor : public Monitor {
 public:
  TorqueMonitor(const libconfig::Setting &settings);
  ~TorqueMonitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:
  jblib::Vec3<double> torque_;
  std::ofstream outfile;
};

#endif  // JAMS_MONITOR_TORQUE_H

