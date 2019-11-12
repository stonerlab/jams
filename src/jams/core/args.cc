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
      if (flag == "--setup-only") {
        program_args.setup_only = true;
        return;
      }

      throw std::runtime_error("Unknown flag \'" + flag + "\'");
    }

    ProgramArgs parse_args(int argc, char **argv) {
      if (argc == 1) throw std::runtime_error("No config file specified");

      ProgramArgs program_args;
      for (auto n = 1; n < argc; ++n) {
        std::string arg(argv[n]);
        trim(arg);

        if (arg_is_flag(arg)) {
          process_flag(arg, program_args);
        } else {
          if (program_args.config_file_path.empty()) {
            program_args.config_file_path = arg;
          } else if (program_args.config_file_patch.empty()) {
            program_args.config_file_patch  = arg;
          } else {
            throw std::runtime_error("too many string arguments on command line");
          }
        }
      }

      return program_args;
    }
}