// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_ENERGY_H
#define JAMS_MONITOR_ENERGY_H

#include <jams/core/monitor.h>
#include <jams/helpers/output.h>

#include <fstream>
#include <string>

class Solver;

class EnergyMonitor : public Monitor {
public:
    explicit EnergyMonitor(const libconfig::Setting &settings);

    ~EnergyMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    static jams::output::TsvWriter make_tsv_writer(const libconfig::Setting &settings);
    jams::output::TsvWriter tsv_;
};

#endif  // JAMS_MONITOR_ENERGY_H
