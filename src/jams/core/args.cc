//
// Created by Joseph Barker on 2019-11-12.
//
#include <string>
#include <ostream>
#include <stdexcept>
#include "jams/core/args.h"
#include "jams/helpers/utils.h"

namespace jams {
    void print_usage(std::ostream& os) {
      os
          << "Usage: jams <config.cfg> [more-configs.cfg ...] [options]\n"
          << "\n"
          << "Options:\n"
          << "  --help                 Show this help\n"
          << "  --version              Print the JAMS version and exit\n"
          << "  --setup-only           Initialise the simulation without running it\n"
          << "  --output=<path>        Write outputs to the given directory\n"
          << "  --name=<name>          Override the simulation name prefix\n"
          << "  --seed=<n>             Override sim.seed with an integer value\n"
          << "  --spins=<path>         Override lattice.spins with a spin-state file\n"
          << "  --temp-directory=<dir> Use the given temporary directory\n"
          << "  --config <libconfig>   Treat the following text as a config string\n";
    }

    bool arg_is_flag(const std::string& arg) {
      return arg.rfind("--", 0) == 0;
    }

    void set_initial_spin_filename(
        const std::string& value,
        const std::string& flag_name,
        ProgramArgs& program_args) {
      if (value.empty()) {
        throw std::runtime_error("Missing value for " + flag_name);
      }
      program_args.initial_spin_filename = value;
    }

    void set_random_seed(
        const std::string& value,
        const std::string& flag_name,
        ProgramArgs& program_args) {
      if (value.empty()) {
        throw std::runtime_error("Missing value for " + flag_name);
      }

      size_t chars_consumed = 0;
      unsigned long parsed_value = 0;
      try {
        parsed_value = std::stoul(value, &chars_consumed);
      } catch (const std::exception&) {
        throw std::runtime_error("Invalid value for " + flag_name + ": " + value);
      }

      if (chars_consumed != value.size()) {
        throw std::runtime_error("Invalid value for " + flag_name + ": " + value);
      }

      program_args.random_seed = parsed_value;
      program_args.random_seed_is_set = true;
    }

    void process_flag(const std::string& flag, ProgramArgs& program_args) {
      if (flag == "--help") {
        program_args.help_only = true;
        return;
      }

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

      if (flag.rfind("--seed=", 0) == 0) {
        set_random_seed(flag.substr(flag.find('=') + 1), "--seed", program_args);
        return;
      }

      if (flag.rfind("--spins=", 0) == 0) {
        set_initial_spin_filename(flag.substr(flag.find('=') + 1), "--spins", program_args);
        return;
      }

      if (flag.rfind("--temp-directory=", 0) == 0) {
        std::string temp_directory = flag.substr(flag.find('=') + 1);
        if (temp_directory.empty()) {
          throw std::runtime_error("Missing value for --temp-directory");
        }
        program_args.temp_directory_path = temp_directory;
        return;
      }

      if (flag.rfind("--temp-dir=", 0) == 0) {
        std::string temp_directory = flag.substr(flag.find('=') + 1);
        if (temp_directory.empty()) {
          throw std::runtime_error("Missing value for --temp-dir");
        }
        program_args.temp_directory_path = temp_directory;
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

        if (arg == "--temp-directory" || arg == "--temp-dir") {
          if (n + 1 >= argc || arg_is_flag(trim(argv[n + 1]))) {
            throw std::runtime_error("Missing value for " + arg);
          }
          ++n;
          program_args.temp_directory_path = trim(argv[n]);
          continue;
        }

        if (arg == "--spins") {
          if (n + 1 >= argc || arg_is_flag(trim(argv[n + 1]))) {
            throw std::runtime_error("Missing value for --spins");
          }
          ++n;
          set_initial_spin_filename(trim(argv[n]), "--spins", program_args);
          continue;
        }

        if (arg == "--seed") {
          if (n + 1 >= argc || arg_is_flag(trim(argv[n + 1]))) {
            throw std::runtime_error("Missing value for --seed");
          }
          ++n;
          set_random_seed(trim(argv[n]), "--seed", program_args);
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
