// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <utility>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/hamiltonian.h"
#include "jams/helpers/output.h"

#include "field.h"

FieldMonitor::FieldMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_(make_tsv_writer())
{}

void FieldMonitor::update(Solver& solver) {
  const auto moments = globals::mus.host_view();

  std::vector<jams::Vec<double, 3>> total_field;
  for (auto &hamiltonian : solver.hamiltonians()) {
    hamiltonian->calculate_fields(solver.time());

    jams::Vec<double, 3> field = {0.0, 0.0, 0.0};
    for (auto i = 0; i < globals::num_spins; ++i) {
      field += jams::Vec<double, 3>{hamiltonian->field(i, 0), hamiltonian->field(i, 1), hamiltonian->field(i, 2)} / moments(i);
    }

    total_field.push_back(field / static_cast<double>(globals::num_spins));
  }

  std::vector<double> values;
  values.reserve(tsv_.num_cols());
  solver.append_monitor_coordinates(values);

  for (const auto& field : total_field) {
    for (auto n = 0; n < 3; ++n) {
      values.push_back(field[n]);
    }
  }

  tsv_.write_row(values);
}

jams::output::TsvWriter FieldMonitor::make_tsv_writer() const {
  auto cols = globals::solver->monitor_coordinate_columns();

  for (const auto& hamiltonian : globals::solver->hamiltonians()) {
    cols.push_back({hamiltonian->name() + "_hx", "T"});
    cols.push_back({hamiltonian->name() + "_hy", "T"});
    cols.push_back({hamiltonian->name() + "_hz", "T"});
  }

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols));
}
