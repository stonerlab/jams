// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_H
#define JAMS_MONITOR_MAGNETISATION_H

#include <fstream>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class MagnetisationMonitor : public Monitor {
 public:
  MagnetisationMonitor(const libconfig::Setting &settings);
  ~MagnetisationMonitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:
  jblib::Array<double, 2> mag;
  std::ofstream outfile;
};

#endif  // JAMS_MONITOR_MAGNETISATION_H
