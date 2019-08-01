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

    void update(Solver *solver) override;
    void post_process() override {};

    bool is_converged() override;

private:
    std::ofstream tsv_file;
    std::string   tsv_header();

    double binder_m2();

    double binder_cumulant();

    jams::MultiArray<Mat3, 1> s_transform_;
    jams::MultiArray<unsigned, 1> material_count_;

    Stats m_stats_;
    Stats m2_stats_;
    Stats m4_stats_;
    std::vector<double> convergence_geweke_m_diagnostic_;
};

#endif  // JAMS_MONITOR_MAGNETISATION_H

