// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SPIN_TEMPERATURE_H
#define JAMS_MONITOR_SPIN_TEMPERATURE_H

#include <fstream>

#include "core/monitor.h"
#include "core/stats.h"

class SpinTemperatureMonitor : public Monitor {
 public:
  SpinTemperatureMonitor(const libconfig::Setting &settings);
  ~SpinTemperatureMonitor();

  void update(Solver * solver);
  bool is_converged();
  std::string name() const {return "spin_temperature";}


 private:
  std::ofstream outfile;

};

#endif  // JAMS_MONITOR_SPIN_TEMPERATURE_H

