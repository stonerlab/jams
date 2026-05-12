// Copyright 2014 Joseph Barker. All rights reserved.

#include "boltzmann.h"

#include <string>
#include <cmath>
#include <vector>

#include "jams/core/solver.h"
#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"

BoltzmannMonitor::BoltzmannMonitor(const libconfig::Setting &settings)
: Monitor(settings),
bins_(90, 0.0),
total_(0),
delay_time_(jams::config_optional<double>(settings, "delay_time", 0.0)/1e-12),
tsv_(jams::output::monitor_filename(name(), "tsv"),
     {{"theta_deg", "degrees", jams::output::ColFmt::Fixed},
      {"probability", "dimensionless"}}) {
}

void BoltzmannMonitor::update(Solver& solver) {
  if (solver.time() < delay_time_) {
    return;
  }
  if (solver.iteration()%output_step_freq_ == 0) {
    const auto spins = globals::s.host_view();
    for (int i = 0; i < globals::num_spins; ++i) {
      double z = spins(i, 2);
      double xy = std::hypot(spins(i, 0), spins(i, 1));
      // Use atan2 rather than acos for accuracy around the poles and to avoid
      // any domain issues if spin is not well normalised
      int round = static_cast<int>(rad_to_deg(std::atan2(xy, z)) * 0.5);
      bins_[round]++;
      total_++;
    }
  }
}

void BoltzmannMonitor::post_process()
{
  if (total_ > 0.0) {
    for (int i = 0; i < 90; ++i) {
      tsv_.write_row_values(i * 2 + 1.0, bins_[i] / total_);
    }
  }
}
