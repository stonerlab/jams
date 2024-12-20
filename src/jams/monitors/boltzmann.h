// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_BOLTZMANN_H
#define JAMS_MONITOR_BOLTZMANN_H

#include <jams/core/monitor.h>

#include <fstream>
#include <vector>

class BoltzmannMonitor : public Monitor {
public:
    explicit BoltzmannMonitor(const libconfig::Setting &settings);
    ~BoltzmannMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    std::vector<double> bins_;
    double total_;
    std::ofstream tsv_file;
};

#endif  // JAMS_MONITOR_BOLTZMANN_H
