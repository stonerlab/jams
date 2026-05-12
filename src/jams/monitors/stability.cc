//
// Created by Joseph Barker on 06/01/2026.
//

#include "jams/monitors/stability.h"

#include <utility>
#include <vector>

#include "jams/containers/vec3.h"
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/helpers/output.h"

StabilityMonitor::StabilityMonitor(const libconfig::Setting& settings)
: Monitor(settings),
  tsv_(make_tsv_writer())
{}


jams::output::TsvWriter StabilityMonitor::make_tsv_writer() const {
  std::vector<jams::output::ColDef> cols = {
      {"time", "picoseconds"},
      {"max_S_err", "dimensionless"},
      {"mean_S_err", "dimensionless"},
      {"rms_S_err", "dimensionless"}};

  return jams::output::TsvWriter(
      jams::output::monitor_filename(name(), "tsv"),
      std::move(cols));
}

void StabilityMonitor::update(Solver& solver)
{
    const auto spins = globals::s.host_view();
    double max_S_err = 0.0;
    double S_err_mean = 0.0;
    double S2_err_mean = 0.0;
    for (auto i = 0; i < globals::num_spins; ++i) {
        const jams::Vec<double, 3> s = {spins(i,0), spins(i, 1), spins(i,2)};
        double S_err = jams::norm_squared(s) - 1;
        S_err_mean += S_err;
        S2_err_mean += S_err * S_err;
        if (std::abs(S_err) > max_S_err) max_S_err = std::abs(S_err);
    }

    S_err_mean /= globals::num_spins;
    S2_err_mean /= globals::num_spins;

    tsv_.write_row_values(solver.time(), max_S_err, S_err_mean, std::sqrt(S2_err_mean));
}
