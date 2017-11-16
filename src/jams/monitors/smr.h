// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SMR_H
#define JAMS_MONITOR_SMR_H

#include <fstream>

#include <libconfig.h++>

#include "jams/core/monitor.h"
#include "jams/core/types.h"

class Solver;

class SMRMonitor : public Monitor {
 public:
  SMRMonitor(const libconfig::Setting &settings);
  ~SMRMonitor();

  void update(Solver * solver);
  bool is_converged() {return false;};
 private:
  std::ofstream outfile;

};

#endif  // JAMS_MONITOR_SMR_H

