// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_VTU_H
#define JAMS_MONITOR_VTU_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

class Solver;

class VtuMonitor : public Monitor {
public:
    explicit VtuMonitor(const libconfig::Setting &settings);

    ~VtuMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    int num_slice_points;
    Vec3 slice_origin;
    Vec3 slice_size;
    std::vector<int> slice_spins;
    jams::MultiArray<int, 1> types_binary_data;
    jams::MultiArray<float, 2> points_binary_data;
    jams::MultiArray<double, 2> spins_binary_data;
};

#endif  // JAMS_MONITOR_VTU_H

