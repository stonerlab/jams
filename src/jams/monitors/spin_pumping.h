// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SPIN_PUMPING_H
#define JAMS_MONITOR_SPIN_PUMPING_H

#include <jams/core/monitor.h>
#include <jams/helpers/output.h>
#include <jams/monitors/spin_grouping.h>

#include <vector>
#include <jams/containers/multiarray.h>

class Solver;

class SpinPumpingMonitor : public Monitor {
public:
    explicit SpinPumpingMonitor(const libconfig::Setting &settings);

    ~SpinPumpingMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

    bool is_updating(const int &iteration) override;

private:
    jams::output::TsvWriter make_tsv_writer() const;
    jams::monitors::SpinGrouping grouping_ = jams::monitors::SpinGrouping::MATERIALS;
    std::vector<jams::monitors::SpinGroup> spin_groups_;
    int precision_ = 8;
    jams::output::TsvWriter tsv_;

    jams::MultiArray<double,2> s_old_;

};

#endif  // JAMS_MONITOR_SPIN_PUMPING_H
