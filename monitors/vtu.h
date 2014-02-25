// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_VTU_H
#define JAMS_MONITOR_VTU_H

#include <fstream>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class VtuMonitor : public Monitor {
 public:
  VtuMonitor(const libconfig::Setting &settings);
  ~VtuMonitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:
};

#endif  // JAMS_MONITOR_VTU_H

