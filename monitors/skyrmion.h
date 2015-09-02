// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SKYRMION_H
#define JAMS_MONITOR_SKYRMION_H

#include <fstream>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class SkyrmionMonitor : public Monitor {
 public:
  SkyrmionMonitor(const libconfig::Setting &settings);
  ~SkyrmionMonitor();

  void update(Solver * solver);
  bool is_converged() { return false; }

 private:
    void create_center_of_mass_mapping();
    void calc_center_of_mass(std::vector<jblib::Vec3<double> > &r_com, const double &threshold);

    std::vector<double> type_norms;
    std::vector<double> thresholds;
    std::ofstream outfile;

    std::vector<jblib::Vec3<double> > tube_x, tube_y;
};

#endif  // JAMS_MONITOR_SKYRMION_H

