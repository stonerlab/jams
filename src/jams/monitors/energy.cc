// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>

#include "jams/core/output.h"
#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"

#include "energy.h"

EnergyMonitor::EnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  std::string name = "_eng.tsv";
  name = seedname+name;
  outfile.open(name.c_str());

  outfile << "time\t";
  for (auto &hh : solver->hamiltonians()) {
    outfile << hh->name() << "\t";
  }
  outfile << std::endl;
}

void EnergyMonitor::update(Solver * solver) {
  outfile << std::setw(12) << std::scientific << solver->time() << "\t";
  outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";

  for (auto &hh : solver->hamiltonians()) {
    outfile << std::setw(21) << std::scientific << std::setprecision(15) << kBohrMagneton * hh->calculate_total_energy() / static_cast<double>(globals::num_spins) << "\t";
  }

  outfile << std::endl;
}

EnergyMonitor::~EnergyMonitor() {
  outfile.close();
}
