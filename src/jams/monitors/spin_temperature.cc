// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <utility>
#include <vector>

#include "jams/helpers/consts.h"
#include "jams/core/globals.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/containers/vec3.h"
#include "jams/helpers/output.h"
#include "spin_temperature.h"

SpinTemperatureMonitor::SpinTemperatureMonitor(const libconfig::Setting &settings)
: Monitor(settings),
  tsv_(make_tsv_writer())
{}

void SpinTemperatureMonitor::update(Solver& solver) {
  const auto spins = globals::s.host_view();
  const auto fields = globals::h.host_view();
  double sum_s_dot_h = 0.0;
  double sum_s_cross_h = 0.0;

  #if HAS_OMP
  #pragma omp parallel for reduction(+:sum_s_cross_h, sum_s_dot_h)
  #endif
  for (auto i = 0; i < globals::num_spins; ++i) {
    const jams::Vec<double, 3> spin = {spins(i,0), spins(i,1), spins(i,2)};
    const jams::Vec<double, 3> field = {fields(i,0), fields(i,1), fields(i,2)};

    sum_s_cross_h += jams::norm_squared(jams::cross(spin, field));
    sum_s_dot_h += jams::dot(spin, field);
  }

  const auto spin_temperature = sum_s_cross_h / (2.0 * kBoltzmannIU * sum_s_dot_h);

  std::vector<double> values;
  values.reserve(tsv_.num_cols());
  solver.append_monitor_coordinates(values);
  values.push_back(solver.physics()->temperature());
  values.push_back(spin_temperature);
  tsv_.write_row(values);
}

jams::output::TsvWriter SpinTemperatureMonitor::make_tsv_writer() const {
  auto cols = globals::solver->monitor_coordinate_columns();
  cols.push_back({"thermostat_T", "K", jams::output::ColFmt::Fixed});
  cols.push_back({"spin_T", "K"});

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols));
}
