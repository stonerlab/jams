//
// Created by Joseph Barker on 2019-11-12.
//
#include <string>
#include "jams/core/args.h"
#include "jams/helpers/utils.h"

namespace jams {
    bool arg_is_flag(const std::string& arg) {
      return arg.rfind("--", 0) == 0;
    }

    void process_flag(const std::string& flag, ProgramArgs& program_args) {
      if (flag == "--version") {
        program_args.version_only = true;
        return;
      }

      if (flag == "--setup-only") {
        program_args.setup_only = true;
        return;
      }

      if (flag.rfind("--output=", 0) == 0) {
        program_args.output_path = flag.substr(flag.find('=') + 1);
        return;
      }

      if (flag.rfind("--name=", 0) == 0) {
        program_args.simulation_name = flag.substr(flag.find('=') + 1);
        return;
      }

      throw std::runtime_error("Unknown flag \'" + flag + "\'");
    }

    ProgramArgs parse_args(int argc, char **argv) {
      if (argc == 1) throw std::runtime_error("No config file specified");

      ProgramArgs program_args;
      for (auto n = 1; n < argc; ++n) {
        std::string arg = trim(argv[n]);

        if (arg_is_flag(arg)) {
          process_flag(arg, program_args);
        } else {
          program_args.config_strings.push_back(arg);
        }
      }

      return program_args;
    }
}