// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SKYRMION_H
#define JAMS_MONITOR_SKYRMION_H

#include <fstream>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class SkyrmionMonitor : public Monitor {
 public:
  SkyrmionMonitor(const libconfig::Setting &settings);
  ~SkyrmionMonitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:
    std::vector<double> type_norms;
    std::ofstream outfile;
};

#endif  // JAMS_MONITOR_SKYRMION_H

