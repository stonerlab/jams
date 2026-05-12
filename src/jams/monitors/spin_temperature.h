// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SPIN_TEMPERATURE_H
#define JAMS_MONITOR_SPIN_TEMPERATURE_H

#include <jams/core/monitor.h>
#include <jams/helpers/output.h>

class Solver;

class SpinTemperatureMonitor : public Monitor {
 public:
  explicit SpinTemperatureMonitor(const libconfig::Setting &settings);
  ~SpinTemperatureMonitor() override = default;

  void update(Solver& solver) override;
    void post_process() override {};

 private:
  jams::output::TsvWriter make_tsv_writer() const;
  jams::output::TsvWriter tsv_;
};

#endif  // JAMS_MONITOR_SPIN_TEMPERATURE_H
