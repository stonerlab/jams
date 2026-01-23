// Copyright 2014 Joseph Barker. All rights reserved.

#include "boltzmann.h"

#include <string>
#include <cmath>

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
tsv_file(jams::output::full_path_filename("blt.tsv")) {
}

void BoltzmannMonitor::update(Solver& solver) {
  if (solver.time() < delay_time_) {
    return;
  }
  if (solver.iteration()%output_step_freq_ == 0) {
    for (int i = 0; i < globals::num_spins; ++i) {
      double z = globals::s(i, 2);
      double xy = std::hypot(globals::s(i, 0), globals::s(i, 1));
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
  tsv_file << "theta_deg probability\n";
  if (total_ > 0.0) {
    for (int i = 0; i < 90; ++i) {
      tsv_file << i * 2 + 1.0 << "\t" << bins_[i] / total_ << "\n";
    }
    tsv_file << "\n" << std::endl;
  }
}
