// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_MAGNETISATION_RATE_H
#define JAMS_MONITOR_MAGNETISATION_RATE_H

#include <fstream>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include "jams/helpers/stats.h"

#include "jblib/containers/array.h"

class MagnetisationRateMonitor : public Monitor {
public:
    explicit MagnetisationRateMonitor(const libconfig::Setting &settings);

    ~MagnetisationRateMonitor() override = default;

    void update(Solver *solver) override;

    bool is_converged() override;

private:
    std::ofstream tsv_file;
    std::string   tsv_header();

    std::vector<int> material_count_;
    Stats magnetisation_stats_;
    bool convergence_is_on_;
    double convergence_tolerance_;
    double convergence_geweke_diagnostic_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_RATE_H

