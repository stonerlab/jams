// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_TORQUE_H
#define JAMS_MONITOR_TORQUE_H

#include <jams/core/monitor.h>
#include <jams/containers/vec3.h>
#include <jams/helpers/output.h>
#include <jams/helpers/stats.h>
#include <jams/monitors/spin_grouping.h>

#include <array>
#include <vector>

class Solver;

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

    void update(Solver& solver) override;
    void post_process() override {};

    ConvergenceStatus convergence_status() override;

private:
    using TorqueComponents = jams::Vec<double, 3>;
    using HamiltonianTorques = std::vector<TorqueComponents>;
    using GroupedTorques = std::vector<HamiltonianTorques>;

    jams::output::TsvWriter make_tsv_writer() const;
    GroupedTorques calculate_torques(Solver& solver);
    TorqueComponents total_group_torque(const HamiltonianTorques& torques) const;
    std::string torque_column_name(
        const jams::monitors::SpinGroup& group,
        const std::string& hamiltonian_name,
        const std::string& component) const;
    std::string convergence_column_name(
        const jams::monitors::SpinGroup& group,
        const std::string& component) const;

    jams::monitors::SpinGrouping grouping_ = jams::monitors::SpinGrouping::NONE;
    std::vector<jams::monitors::SpinGroup> spin_groups_;
    int precision_ = 8;

    std::vector<std::array<Stats, 3>> torque_stats_;
    std::vector<TorqueComponents> convergence_geweke_diagnostic_;
    jams::output::TsvWriter tsv_;
};

#endif  // JAMS_MONITOR_TORQUE_H
