// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_BINARY_H
#define JAMS_MONITOR_BINARY_H

#include <jams/core/monitor.h>

class BinaryMonitor : public Monitor {
 public:
  explicit BinaryMonitor(const libconfig::Setting &settings);
    ~BinaryMonitor() override = default;

  void update(Solver& solver) override;
  void post_process() override {};

 private:
    bool is_file_overwrite_mode;
};

#endif  // JAMS_MONITOR_BINARY_H

