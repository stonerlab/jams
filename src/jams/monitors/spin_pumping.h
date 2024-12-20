// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_SPIN_PUMPING_H
#define JAMS_MONITOR_SPIN_PUMPING_H

#include <jams/core/monitor.h>

#include <fstream>
#include <string>
#include <vector>
#include <jams/containers/multiarray.h>

class Solver;

class SpinPumpingMonitor : public Monitor {
public:
    explicit SpinPumpingMonitor(const libconfig::Setting &settings);

    ~SpinPumpingMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

    bool is_updating(const int &iteration) override;

private:
    std::string      tsv_header();

    std::ofstream    tsv_file_;
    std::vector<int> material_count_;

    jams::MultiArray<double,2> s_old_;

};

#endif  // JAMS_MONITOR_SPIN_PUMPING_H
