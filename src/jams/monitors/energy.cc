// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/helpers/output.h"

#include "energy.h"

#include "jams/interface/config.h"


EnergyMonitor::EnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_file(jams::output::full_path_filename("eng.tsv")) {
  output_precision_ = jams::config_optional<int>(settings, "precision", 8);
  tsv_file << tsv_header();
}

void EnergyMonitor::update(Solver& solver) {

  tsv_file << jams::fmt::sci(output_precision_) << solver.time();

  for (auto &hamiltonian : solver.hamiltonians()) {
    auto energy = hamiltonian->calculate_total_energy(solver.time());
    tsv_file << jams::fmt::sci(output_precision_) << energy;
  }

  tsv_file << std::endl;
}

std::string EnergyMonitor::tsv_header() {
  std::vector<jams::output::ColDef> cols;

  cols.push_back({"time", "picoseconds"});
  for (const auto& h : globals::solver->hamiltonians()) {
    cols.push_back({h->name(), "meV"});
  }

  std::string units_line = jams::output::make_json_units_string(cols);
  std::string header_line = jams::output::make_tsv_header_row(cols, output_precision_);

  return units_line + header_line;
}
