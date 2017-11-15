// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_TORQUE_H
#define JAMS_MONITOR_TORQUE_H

#include <fstream>
#include <array>

#include <libconfig.h++>

#include "jams/core/monitor.h"
#include "jams/core/types.h"

class Solver;
class Stats;

class TorqueMonitor : public Monitor {
 public:
  TorqueMonitor(const libconfig::Setting &settings);
  ~TorqueMonitor();

  void update(Solver * solver);
  bool is_converged();
  std::string name() const {return "torque";}

 private:

  void open_outfile();

  std::ofstream outfile;
  std::array<Stats,3> torque_stats_;
  std::array<double,3> convergence_geweke_diagnostic_;
};

#endif  // JAMS_MONITOR_TORQUE_H

