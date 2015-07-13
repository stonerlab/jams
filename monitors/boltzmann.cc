// Copyright 2014 Joseph Barker. All rights reserved.

#include "monitors/boltzmann.h"

#include <string>

#include "core/globals.h"
#include "core/maths.h"

BoltzmannMonitor::BoltzmannMonitor(const libconfig::Setting &settings)
: Monitor(settings),
bins(0),
total(0),
outfile() {
  ::output.write("Initialising Boltzmann monitor\n");

  std::string name = "_blt.dat";
  name = seedname+name;
  outfile.open(name.c_str());

  bins.resize(36);
  for (int i = 0; i < 36; ++i) {
    bins(i) = 0.0;
  }
}

void BoltzmannMonitor::update(const Solver * const solver) {
  using namespace globals;

  if (solver->iteration()%output_step_freq_ == 0) {
    for (int i = 0; i < num_spins; ++i) {
      int round = static_cast<int>(rad_to_deg(acos(s(i, 2)))*0.2);
      bins(round)++;
      total++;
    }

    if (total > 0.0) {
      for (int i = 0; i < 36; ++i) {
        outfile << i*5+2.5 << "\t" << bins(i)/total << "\n";
      }
      outfile << "\n" << std::endl;
    }
  }
}

BoltzmannMonitor::~BoltzmannMonitor() {
  outfile.close();
}
