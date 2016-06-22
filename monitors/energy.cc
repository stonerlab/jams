// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/hamiltonian.h"

#include "monitors/energy.h"

EnergyMonitor::EnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Energy monitor...\n");

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
      outfile << std::setw(20) << std::scientific << (*it)->calculate_total_energy() / static_cast<double>(num_spins) << "\t";
    }
  }

  outfile << std::endl;
}

EnergyMonitor::~EnergyMonitor() {
  outfile.close();
}
