// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_H
#define JAMS_MONITOR_MAGNETISATION_H

#include <jams/helpers/output.h>
#include <jams/core/monitor.h>
#include <jams/monitors/spin_grouping.h>

#include <fstream>
#include <vector>
#include <string>

class Solver;

class MagnetisationMonitor : public Monitor {
public:
    explicit MagnetisationMonitor(const libconfig::Setting &settings);

    ~MagnetisationMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    jams::output::TsvWriter make_tsv_writer(const libconfig::Setting &settings);

    jams::monitors::SpinGrouping grouping_ = jams::monitors::SpinGrouping::MATERIALS;
    bool normalize_magnetisation_ = true;
    std::vector<jams::monitors::SpinGroup> spin_groups_;

    jams::output::TsvWriter tsv_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_H
