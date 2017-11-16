// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SPIN_TEMPERATURE_H
#define JAMS_MONITOR_SPIN_TEMPERATURE_H

#include <fstream>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

class Solver;

class SpinTemperatureMonitor : public Monitor {
 public:
  SpinTemperatureMonitor(const libconfig::Setting &settings);
  ~SpinTemperatureMonitor();

  void update(Solver * solver);
  bool is_converged();

 private:
  std::ofstream outfile;

};

#endif  // JAMS_MONITOR_SPIN_TEMPERATURE_H

