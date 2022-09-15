// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SMR_H
#define JAMS_MONITOR_SMR_H

#include <fstream>

#include <libconfig.h++>

#include "jams/core/monitor.h"
#include "jams/core/types.h"

class Solver;

class SMRMonitor : public Monitor {
public:
    explicit SMRMonitor(const libconfig::Setting &settings);

    ~SMRMonitor() override = default;

    void update(Solver *solver) override;
    void post_process() override {};

private:
    std::ofstream tsv_file;

    std::string tsv_header();

};

#endif  // JAMS_MONITOR_SMR_H

