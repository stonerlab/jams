// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_H
#define JAMS_MONITOR_MAGNETISATION_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include "jams/helpers/stats.h"

#include "jblib/containers/array.h"

class Solver;

class MagnetisationMonitor : public Monitor {
 public:
  MagnetisationMonitor(const libconfig::Setting &settings);
  ~MagnetisationMonitor();

  void update(Solver * solver);
  bool is_converged();


 private:
  double binder_m2();
  double binder_cumulant();

  jblib::Array<double, 2> mag;
  jblib::Array<double, 2> s_transform_;

  std::ofstream outfile;

  Stats m_stats_;
  Stats m2_stats_;
  Stats m4_stats_;
  std::vector<double> convergence_geweke_m_diagnostic_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_H

