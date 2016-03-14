// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_H
#define JAMS_MONITOR_MAGNETISATION_H

#include <fstream>

#include "core/monitor.h"
#include "core/stats.h"

#include "jblib/containers/array.h"

class MagnetisationMonitor : public Monitor {
 public:
  MagnetisationMonitor(const libconfig::Setting &settings);
  ~MagnetisationMonitor();

  void update(Solver * solver);
  bool is_converged();
  std::string name() const {return "magnetisation";}


 private:
  jblib::Array<double, 2> mag;
  std::ofstream outfile;

  Stats mt_stats_;
  Stats mz_stats_;
  bool convergence_is_on_;
  double convergence_tolerance_;
  double convergence_geweke_mt_diagnostic_;
  double convergence_geweke_mz_diagnostic_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_H

