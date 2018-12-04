// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_XYZ_H
#define JAMS_MONITOR_XYZ_H

#include <fstream>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

#include "jblib/containers/array.h"

class Solver;

class XyzMonitor : public Monitor {
public:
    explicit XyzMonitor(const libconfig::Setting &settings);

    ~XyzMonitor() override = default;

    void update(Solver *solver) override;

    bool is_converged() override { return false; }


private:
    Vec3 slice_origin;
    Vec3 slice_size;
    std::vector<int> slice_spins;
};

#endif  // JAMS_MONITOR_XYZ_H
