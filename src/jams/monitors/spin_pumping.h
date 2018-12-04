// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SPIN_PUMPING_H
#define JAMS_MONITOR_SPIN_PUMPING_H

#include <fstream>

#include <libconfig.h++>

#include "jams/helpers/stats.h"
#include "jams/core/monitor.h"

class Solver;

class SpinPumpingMonitor : public Monitor {
public:
    explicit SpinPumpingMonitor(const libconfig::Setting &settings);

    ~SpinPumpingMonitor() override = default;

    void update(Solver *solver) override;

    bool is_converged() override {return false;}

private:
    std::ofstream tsv_file;
    std::string   tsv_header();
};

#endif  // JAMS_MONITOR_SPIN_PUMPING_H