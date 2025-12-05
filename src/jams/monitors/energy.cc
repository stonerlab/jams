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
  tsv_file_(jams::output::full_path_filename("eng.tsv")),
  output_precision_(jams::config_optional<int>(settings, "precision", 8))
{

  tsv_cols_.push_back({"time", "picoseconds"});
  for (const auto& h : globals::solver->hamiltonians()) {
    tsv_cols_.push_back({h->name(), "meV"});
  }

  tsv_file_ <<  jams::output::make_json_units_string(tsv_cols_);
  tsv_file_ <<  jams::output::make_tsv_header_row(tsv_cols_, output_precision_);

}

void EnergyMonitor::update(Solver& solver) {

  std::vector<double> values;
  values.reserve(tsv_cols_.size());

  values.push_back(solver.time());

  for (auto &h : solver.hamiltonians()) {
    values.push_back(h->calculate_total_energy(solver.time()));
  }

  jams::output::write_tsv_row(tsv_file_, tsv_cols_, values, output_precision_);
}

