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

#include "jams/helpers/container_utils.h"
#include "jams/interface/config.h"


EnergyMonitor::EnergyMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_(make_tsv_writer(settings))
{}

void EnergyMonitor::update(Solver& solver) {
  auto values = make_reserved<double>(tsv_.num_cols());

  values.push_back(solver.time());
  for (auto &h : solver.hamiltonians()) {
    values.push_back(h->calculate_total_energy(solver.time()));
  }

  tsv_.write_row(values);
}


jams::output::TsvWriter EnergyMonitor::make_tsv_writer(const libconfig::Setting &settings) {
  auto precision = jams::config_optional<int>(settings, "precision", 8);

  std::vector<jams::output::ColDef> cols;
  cols.push_back({"time", "picoseconds", jams::output::ColFmt::Scientific});

  for (const auto& h : globals::solver->hamiltonians()) {
    cols.push_back({h->name(), "meV", jams::output::ColFmt::Scientific});
  }

  return jams::output::TsvWriter(
      jams::output::full_path_filename("eng.tsv"),
      std::move(cols),
      precision
  );
}

