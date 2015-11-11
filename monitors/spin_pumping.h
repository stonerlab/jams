// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SPIN_PUMPING_H
#define JAMS_MONITOR_SPIN_PUMPING_H

#include <fstream>

#include "core/stats.h"
#include "core/monitor.h"
#include "jblib/containers/array.h"

class SpinPumpingMonitor : public Monitor{
 public:
  SpinPumpingMonitor(const libconfig::Setting &settings);
  ~SpinPumpingMonitor();

void update(Solver * solver);
bool is_converged();
std::string name() const {return "boltzmann";}


 private:
  std::ofstream w_dist_file;
  std::ofstream iz_dist_file;
  std::ofstream iz_mean_file;

  Stats convergence_stats_;
  bool convergence_is_on_;
  double convergence_tolerance_;
  double convergence_geweke_diagnostic_;
};

#endif  // JAMS_MONITOR_SPIN_PUMPING_H
