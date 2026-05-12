// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SMR_H
#define JAMS_MONITOR_SMR_H

#include <jams/core/monitor.h>
#include <jams/monitors/spin_grouping.h>

#include <fstream>
#include <string>
#include <vector>

class Solver;

class SMRMonitor : public Monitor {
public:
    explicit SMRMonitor(const libconfig::Setting &settings);

    ~SMRMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    std::ofstream tsv_file;
    std::vector<jams::monitors::SpinGroup> spin_groups_;

    std::string tsv_header();

};

#endif  // JAMS_MONITOR_SMR_H
