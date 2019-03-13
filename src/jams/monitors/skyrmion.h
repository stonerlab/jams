// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SKYRMION_H
#define JAMS_MONITOR_SKYRMION_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"

class SkyrmionMonitor : public Monitor {
 public:
  SkyrmionMonitor(const libconfig::Setting &settings);
  ~SkyrmionMonitor();

  void update(Solver * solver) override;
    void post_process() override {};

    bool is_converged() override { return false; }

 private:
    std::string tsv_header();

    void create_center_of_mass_mapping();
    void calc_center_of_mass(std::vector<Vec3 > &r_com, const double &threshold);

    std::vector<double> type_norms;
    std::vector<double> thresholds;
    std::ofstream outfile;

    std::vector<Vec3 > tube_x, tube_y;
};

#endif  // JAMS_MONITOR_SKYRMION_H

