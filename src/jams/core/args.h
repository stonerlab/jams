//
// Created by Joseph Barker on 2019-11-12.
//

#ifndef JAMS_ARGS_H
#define JAMS_ARGS_H

#include <string>

namespace jams {

    struct ProgramArgs {
        bool        setup_only        = false;
        std::string config_file_path  = "";
        std::string config_file_patch = "";
        std::string output_path = "";
        std::string simulation_name = "";
    };

    ProgramArgs parse_args(int argc, char **argv);
}

#endif //JAMS_ARGS_H
