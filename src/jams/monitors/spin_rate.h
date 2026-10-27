#ifndef JAMS_MONITORS_SPIN_RATE_H
#define JAMS_MONITORS_SPIN_RATE_H

#include <optional>
#include "jams/core/monitor.h"
#include "jams/helpers/output.h"
#include "jams/helpers/scalar_threshold.h"

class SpinRateMonitor : public Monitor {
 public:
  explicit SpinRateMonitor(const libconfig::Setting& settings);
  void update(Solver& solver) override;
  void post_process() override {}

 private:
  std::optional<jams::ScalarThreshold> threshold_;
  jams::output::TsvWriter tsv_;
};

#endif
