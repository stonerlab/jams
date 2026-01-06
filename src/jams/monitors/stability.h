//
// Created by Joseph Barker on 06/01/2026.
//

#ifndef JAMS_MONITOR_STABILITY_H
#define JAMS_MONITOR_STABILITY_H
#include <fstream>

#include "jams/core/monitor.h"

class Solver;

class StabilityMonitor : public Monitor {
public:
    explicit StabilityMonitor(const libconfig::Setting &settings);

    void update(Solver& solver) override;
    void post_process() override {};

private:
    std::ofstream tsv_file;
    std::string   tsv_header();
};

#endif //JAMS_MONITOR_STABILITY_H