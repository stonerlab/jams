// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_FIELD_H
#define JAMS_MONITOR_FIELD_H

#include <fstream>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/solver.h"
#include "jams/core/monitor.h"

class FieldMonitor : public Monitor {
public:
    explicit FieldMonitor(const libconfig::Setting &settings);

    ~FieldMonitor() override = default;

    void update(Solver& solver) override;
    void post_process() override {};

private:
    std::ofstream tsv_file;
    std::string   tsv_header();

};

#endif  // JAMS_MONITOR_FIELD_H
