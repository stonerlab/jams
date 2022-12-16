// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_H
#define JAMS_MONITOR_MAGNETISATION_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include "jams/helpers/stats.h"

class Solver;

class MagnetisationMonitor : public Monitor {
public:
    explicit MagnetisationMonitor(const libconfig::Setting &settings);

    ~MagnetisationMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    enum class Grouping {
        MATERIALS,
        POSITIONS
    };

    Grouping grouping_ = Grouping::MATERIALS;
    bool normalize_magnetisation_ = true;
    std::ofstream tsv_file;
    std::string   tsv_header();

    std::vector<jams::MultiArray<int,1>> group_spin_indicies_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_H

