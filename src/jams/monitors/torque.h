// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_TORQUE_H
#define JAMS_MONITOR_TORQUE_H

#include <fstream>
#include <array>

#include <libconfig.h++>

#include "jams/core/monitor.h"
#include "jams/core/types.h"

class Solver;
class Stats;

///
/// Calculates the average torque per spin from each Hamiltonian term as
/// @f[
/// \mathbf{\tau} = \frac{1}{N}\sum_i^N \mathbf{S}_i \times \mathbf{H}_i
/// @f]
///
class TorqueMonitor : public Monitor {
public:
    explicit TorqueMonitor(const libconfig::Setting &settings);

    ~TorqueMonitor() = default;

    void update(Solver *solver) override;
    void post_process() override {};

    bool is_converged() override;

private:
    std::ofstream tsv_file;
    std::string   tsv_header();

    std::array<Stats, 3> torque_stats_;
    Vec3 convergence_geweke_diagnostic_;
};

#endif  // JAMS_MONITOR_TORQUE_H

