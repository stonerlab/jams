// Copyright 2017 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_JAMS_H
#define JAMS_CORE_JAMS_H

#include <string>
#include <libconfig.h++>
#include <random>

#include "jams/helpers/defaults.h"

namespace jams {
    struct Simulation {
        std::string name;
        bool verbose = jams::defaults::sim_verbose_output;

        std::string config_file_name;
        std::string config_patch_string;

        std::string   random_state;
        unsigned long random_seed;
    };
}

void jams_initialize(int argc, char **argv);
void jams_patch_config(const std::string &patch_string);
void jams_run();
void jams_finish();
void jams_global_initializer(const libconfig::Setting &settings);

#endif
