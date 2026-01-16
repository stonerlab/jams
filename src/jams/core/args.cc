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

    void add_config_input(const std::string& value, bool force_string, ProgramArgs& program_args) {
      program_args.config_inputs.push_back({value, force_string});
    }

    std::string parse_config_value(int argc, char **argv, int &index) {
      std::string combined = trim(argv[index]);
      while (index + 1 < argc) {
        std::string next_arg = trim(argv[index + 1]);
        if (arg_is_flag(next_arg)) {
          break;
        }
        combined += " " + next_arg;
        ++index;
      }
      return combined;
    }

    ProgramArgs parse_args(int argc, char **argv) {
      if (argc == 1) throw std::runtime_error("No config specified");

      ProgramArgs program_args;
      for (int n = 1; n < argc; ++n) {
        std::string arg = trim(argv[n]);

        if (arg == "--config") {
          if (n + 1 >= argc || arg_is_flag(trim(argv[n + 1]))) {
            throw std::runtime_error("Missing value for --config");
          }
          ++n;
          std::string config_value = parse_config_value(argc, argv, n);
          add_config_input(config_value, true, program_args);
          continue;
        }

        if (arg.rfind("--config=", 0) == 0) {
          std::string config_value = arg.substr(arg.find('=') + 1);
          if (config_value.empty()) {
            throw std::runtime_error("Missing value for --config");
          }
          add_config_input(config_value, true, program_args);
          continue;
        }

        if (arg_is_flag(arg)) {
          process_flag(arg, program_args);
          continue;
        }

        add_config_input(arg, false, program_args);
      }

      return program_args;
    }
}
