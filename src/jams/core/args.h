//
// Created by Joseph Barker on 2019-11-12.
//

#ifndef JAMS_ARGS_H
#define JAMS_ARGS_H

#include <string>
#include <vector>

namespace jams {

    struct ProgramArgs {
        bool        setup_only        = false;
        std::string output_path = "";
        std::string simulation_name = "";

        // a vector of filenames or patch strings to assemble to config
        std::vector<std::string> config_strings;
    };

    ProgramArgs parse_args(int argc, char **argv);
}

#endif //JAMS_ARGS_H
