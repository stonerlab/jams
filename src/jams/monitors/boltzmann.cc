// Copyright 2014 Joseph Barker. All rights reserved.

#include "boltzmann.h"

#include <string>
#include <cmath>

#include "jams/core/solver.h"
#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/output.h"

BoltzmannMonitor::BoltzmannMonitor(const libconfig::Setting &settings)
: Monitor(settings),
bins_(36, 0.0),
total_(0),
tsv_file(jams::output::full_path_filename("blt.tsv")) {
}

void BoltzmannMonitor::update(Solver * solver) {
  if (solver->iteration()%output_step_freq_ == 0) {
    for (int i = 0; i < globals::num_spins; ++i) {
      int round = static_cast<int>(rad_to_deg(acos(globals::s(i, 2)))*0.2);
      bins_[round]++;
      total_++;
    }

    if (total_ > 0.0) {
      for (int i = 0; i < 36; ++i) {
        tsv_file << i*5+2.5 << "\t" << bins_[i]/total_ << "\n";
      }
      tsv_file << "\n" << std::endl;
    }
  }
}