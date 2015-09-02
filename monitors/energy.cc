// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>

#include "core/globals.h"
#include "core/lattice.h"

#include "monitors/energy.h"

EnergyMonitor::EnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Energy monitor...\n");

  std::string name = "_eng.dat";
  name = seedname+name;
  outfile.open(name.c_str());

  outfile << "# time (s) | e_tot | e1_s | e1_t | e2_s | e2_t | e4_s "
    << std::endl;
}

void EnergyMonitor::update(Solver * solver) {
  using namespace globals;

  if (solver->iteration()%output_step_freq_ == 0) {
  // double e1_s = 0.0, e1_t = 0.0, e2_s = 0.0, e2_t = 0.0, e4_s = 0.0;

  // solver->compute_total_energy(e1_s, e1_t, e2_s, e2_t, e4_s);

  //   outfile << solver->time();

  //     outfile << "\t" << e1_s+e1_t+e2_s+e2_t+e4_s;

  //     outfile << "\t" << e1_s;
  //     outfile << "\t" << e1_t;
  //     outfile << "\t" << e2_s;
  //     outfile << "\t" << e2_t;
  //     outfile << "\t" << e4_s;

#ifdef NDEBUG
  outfile << "\n";
#else
  outfile << std::endl;
#endif
  }
}

EnergyMonitor::~EnergyMonitor() {
  outfile.close();
}
