// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>

#include "jams/core/output.h"
#include "jams/core/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"

#include "jams/monitors/energy.h"

EnergyMonitor::EnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output->write("\nInitialising Energy monitor...\n");

  std::string name = "_eng.tsv";
  name = seedname+name;
  outfile.open(name.c_str());

  outfile << "time\t";
  for (auto it = solver->hamiltonians().begin() ; it != solver->hamiltonians().end(); ++it) {
    outfile << (*it)->name() << "\t";
  }
  outfile << std::endl;
}

void EnergyMonitor::update(Solver * solver) {
  using namespace globals;

  outfile << std::setw(12) << std::scientific << solver->time() << "\t";
  outfile << std::setw(12) << std::fixed << solver->physics()->temperature() << "\t";

  if (solver->iteration()%output_step_freq_ == 0) {
    for (auto it = solver->hamiltonians().begin() ; it != solver->hamiltonians().end(); ++it) {
      outfile << std::setw(21) << std::scientific << std::setprecision(15) << kBohrMagneton * (*it)->calculate_total_energy() / static_cast<double>(num_spins) << "\t";
    }
  }

  outfile << std::endl;
}

EnergyMonitor::~EnergyMonitor() {
  outfile.close();
}
