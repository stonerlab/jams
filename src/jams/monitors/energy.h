// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_ENERGY_H
#define JAMS_MONITOR_ENERGY_H

#include <fstream>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/solver.h"
#include "jams/core/monitor.h"

class EnergyMonitor : public Monitor {
 public:
  EnergyMonitor(const libconfig::Setting &settings);

  ~EnergyMonitor();
  void update(Solver * solver);
  bool is_converged() { return false; }

 private:
  std::ofstream outfile;
};

#endif  // JAMS_MONITOR_ENERGY_H
