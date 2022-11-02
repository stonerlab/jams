// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_ENERGY_H
#define JAMS_MONITOR_ENERGY_H

#include <fstream>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/solver.h"
#include "jams/core/monitor.h"

class EnergyMonitor : public Monitor {
public:
    explicit EnergyMonitor(const libconfig::Setting &settings);

    ~EnergyMonitor() override = default;

    void update(Solver *solver) override;
    void post_process() override {};

private:
    std::ofstream tsv_file;
    std::string   tsv_header();

};

#endif  // JAMS_MONITOR_ENERGY_H
