// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SMR_H
#define JAMS_MONITOR_SMR_H

#include <fstream>

#include "jams/core/monitor.h"
#include "jams/core/stats.h"

#include "jblib/containers/array.h"

class SMRMonitor : public Monitor {
 public:
  SMRMonitor(const libconfig::Setting &settings);
  ~SMRMonitor();

  void update(Solver * solver);
  bool is_converged() {return false;};
  std::string name() const {return "smr";}

 private:
  std::ofstream outfile;

};

#endif  // JAMS_MONITOR_SMR_H

