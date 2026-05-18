// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SKYRMION_H
#define JAMS_MONITOR_SKYRMION_H

#include <jams/core/types.h>
#include <jams/core/monitor.h>
#include <jams/helpers/output.h>

#include <vector>

class Solver;

class SkyrmionMonitor : public Monitor {
 public:
  SkyrmionMonitor(const libconfig::Setting &settings);
  ~SkyrmionMonitor();

  void update(Solver& solver) override;
    void post_process() override {};

 private:
    jams::output::TsvWriter make_tsv_writer() const;

    void create_center_of_mass_mapping();
    void calc_center_of_mass(std::vector<jams::Vec<double, 3> > &r_com, const double &threshold);

    std::vector<double> type_norms;
    std::vector<double> thresholds;
    jams::output::TsvWriter tsv_;

    std::vector<jams::Vec<double, 3> > tube_x, tube_y;
};

#endif  // JAMS_MONITOR_SKYRMION_H
