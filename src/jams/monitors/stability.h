//
// Created by Joseph Barker on 06/01/2026.
//

#ifndef JAMS_MONITOR_STABILITY_H
#define JAMS_MONITOR_STABILITY_H

#include "jams/core/monitor.h"
#include "jams/helpers/output.h"

class Solver;

class StabilityMonitor : public Monitor {
public:
    explicit StabilityMonitor(const libconfig::Setting &settings);

    void update(Solver& solver) override;
    void post_process() override {};

private:
    jams::output::TsvWriter make_tsv_writer() const;
    jams::output::TsvWriter tsv_;
};

#endif //JAMS_MONITOR_STABILITY_H
