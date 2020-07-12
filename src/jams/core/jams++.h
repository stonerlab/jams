// Copyright 2017 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_JAMS_H
#define JAMS_CORE_JAMS_H

#include <string>
#include <libconfig.h++>

#include "jams/core/args.h"
#include "jams/helpers/defaults.h"

namespace jams {
    struct Simulation {
        bool verbose = jams::defaults::sim_verbose_output;

        std::string   random_state;
        unsigned long random_seed;
    };

    void initialize_simulation(const jams::ProgramArgs& program_args);
    void run_simulation();
    void cleanup_simulation();
    void global_initializer(const libconfig::Setting &settings);
}



#endif
