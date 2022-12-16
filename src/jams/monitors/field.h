// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_FIELD_H
#define JAMS_MONITOR_FIELD_H

#include <jams/core/monitor.h>

#include <fstream>
#include <string>

class Solver;

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
