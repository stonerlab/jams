// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_ENERGY_H
#define JAMS_MONITOR_ENERGY_H

#include <fstream>

#include "core/monitor.h"

class EnergyMonitor : public Monitor {
 public:
  EnergyMonitor() {}

  ~EnergyMonitor();

  void initialise();
  void run();
  void write(Solver *solver);
  void initConvergence(ConvergenceType type, const double meanTol,
    const double devTol);
  bool checkConvergence();
 private:
  std::ofstream outfile;
};

#endif  // JAMS_MONITOR_ENERGY_H
