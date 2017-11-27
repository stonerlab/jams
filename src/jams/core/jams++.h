// Copyright 2017 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_JAMS_H
#define JAMS_CORE_JAMS_H

#include <ctime>
#include <string>

#include <libconfig.h++>

#include "jams/helpers/defaults.h"

namespace jams {
    struct Simulation {
        std::string name;
        bool verbose = jams::default_sim_verbose_output;

        std::string log_file_name;
        std::string config_file_name;
        std::string config_patch_string;

        int random_seed = time(nullptr);
    };
}

int jams_initialize(int argc, char **argv);
void jams_patch_config(const std::string &patch_string);
void jams_run();
void jams_finish();
void jams_global_initializer(const libconfig::Setting &settings);

#endif