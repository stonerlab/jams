// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"

#include "energy.h"

EnergyMonitor::EnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  tsv_file.open(seedname + "_eng.tsv");
  tsv_file.setf(std::ios::right);
  tsv_file << tsv_header();
}

void EnergyMonitor::update(Solver * solver) {
  tsv_file.width(12);

  tsv_file << std::scientific << solver->time() << "\t";

  for (auto &hamiltonian : solver->hamiltonians()) {
    auto energy = kBohrMagneton * hamiltonian->calculate_total_energy() / static_cast<double>(globals::num_spins);
    tsv_file << std::scientific << std::setprecision(15) << energy << "\t";
  }

  tsv_file << std::endl;
}

std::string EnergyMonitor::tsv_header() {
  std::stringstream ss;
  ss.width(12);

  ss << "time\t";
  for (auto &hamiltonian : solver->hamiltonians()) {
    ss << hamiltonian->name() << "_e\t";
  }

  ss << std::endl;

  return ss.str();
}
