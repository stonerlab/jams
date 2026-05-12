// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/helpers/maths.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/output.h"

#include "jams/monitors/magnetisation.h"
#include "jams/helpers/spinops.h"
#include "jams/helpers/array_ops.h"
#include "jams/helpers/container_utils.h"

MagnetisationMonitor::MagnetisationMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_(make_tsv_writer(settings))
{}

void MagnetisationMonitor::update(Solver& solver) {
  const auto& spins = globals::s;
  const auto& moments = globals::mus;
  auto values = make_reserved<double>(tsv_.num_cols());

  values.push_back(solver.time());

  for (const auto& group : spin_groups_) {
    if (group.empty()) {
      values.push_back(0.0);
      values.push_back(0.0);
      values.push_back(0.0);
      values.push_back(0.0);
      continue;
    }

    jams::Vec<double, 3> mag = jams::sum_spins_moments(spins, moments, group.indices_array());
    double normalising_factor = 1.0;
    if (normalize_magnetisation_) {
      normalising_factor = 1.0 / jams::scalar_field_indexed_reduce(moments, group.indices_array());
    } else {
      // internally we use meV T^-1 for mus so convert back to Bohr magneton
      normalising_factor = 1.0 / kBohrMagnetonIU;
    }

    values.push_back(mag[0] * normalising_factor);
    values.push_back(mag[1] * normalising_factor);
    values.push_back(mag[2] * normalising_factor);
    values.push_back(jams::norm(mag) * normalising_factor);
  }

  tsv_.write_row(values);
}


jams::output::TsvWriter MagnetisationMonitor::make_tsv_writer(const libconfig::Setting &settings) {
  grouping_ = jams::monitors::parse_spin_grouping(settings, "materials", "magnetisation");
  spin_groups_ = jams::monitors::make_spin_groups(grouping_);

  // should the magnetisation be normalised to 1 or be in units of muB
  normalize_magnetisation_ = jams::config_optional<bool>(settings, "normalize", true);

  auto precision = jams::config_optional<int>(settings, "precision", 8);
  std::vector<jams::output::ColDef> cols;

  std::string mag_unit = "dimensionless";
  if (!normalize_magnetisation_) {
    mag_unit = "bohr magnetons";
  }

  cols.push_back({"time", "picoseconds"});

  for (const auto& group : spin_groups_) {
    for (const auto& component : {"mx", "my", "mz", "m"}) {
      cols.push_back({
          jams::monitors::grouped_column_name(grouping_, group.name, component),
          mag_unit});
    }
  }

  return jams::output::TsvWriter(
    jams::output::monitor_filename(name(), "tsv"),
    std::move(cols),
    precision
  );
}
