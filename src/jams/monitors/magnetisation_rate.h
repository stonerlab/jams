// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_RATE_H
#define JAMS_MONITOR_MAGNETISATION_RATE_H

#include <jams/core/monitor.h>
#include <jams/helpers/stats.h>
#include <jams/monitors/spin_grouping.h>

#include <fstream>
#include <string>
#include <vector>

class Solver;

class MagnetisationRateMonitor : public Monitor {
public:
    explicit MagnetisationRateMonitor(const libconfig::Setting &settings);

    ~MagnetisationRateMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

    ConvergenceStatus convergence_status() override;

private:
    std::ofstream tsv_file;
    std::string   tsv_header();

    std::vector<jams::monitors::SpinGroup> spin_groups_;
    Stats magnetisation_stats_;
    double convergence_geweke_diagnostic_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_RATE_H
