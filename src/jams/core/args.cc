//
// Created by Joseph Barker on 2019-11-12.
//
#include <string>
#include <ostream>
#include <stdexcept>
#include "jams/core/args.h"
#include "jams/helpers/utils.h"

namespace jams {
    bool arg_is_flag(const std::string& arg);

    bool flag_matches(const std::string& arg, const std::string& flag_name) {
      return arg == flag_name || arg.rfind(flag_name + "=", 0) == 0;
    }

    std::string flag_value_from_equals_form(const std::string& arg, const std::string& flag_name) {
      return arg.substr(flag_name.size() + 1);
    }

    std::string consume_flag_value(
        int argc,
        char **argv,
        int &index,
        const std::string& arg,
        const std::string& flag_name) {
      if (arg == flag_name) {
        if (index + 1 >= argc || arg_is_flag(trim(argv[index + 1]))) {
          throw std::runtime_error("Missing value for " + flag_name);
        }
        ++index;
        return trim(argv[index]);
      }

      if (arg.rfind(flag_name + "=", 0) == 0) {
        auto value = flag_value_from_equals_form(arg, flag_name);
        if (value.empty()) {
          throw std::runtime_error("Missing value for " + flag_name);
        }
        return value;
      }

      return "";
    }

    void print_usage(std::ostream& os) {
      os
          << "Usage: jams <config.cfg> [more-configs.cfg ...] [options]\n"
          << "\n"
          << "Options:\n"
          << "  --help                 Show this help\n"
          << "  --version              Print the JAMS version and exit\n"
          << "  --validate-config      Parse and validate the config, then exit\n"
          << "  --setup-only           Initialise the simulation without running it\n"
          << "  --output=<path>        Write outputs to the given directory\n"
          << "  --log=<path>           Redirect stdout/stderr messages to the given file\n"
          << "  --name=<name>          Override the simulation name prefix\n"
          << "  --seed=<n>             Override sim.seed with an integer value\n"
          << "  --verbose              Override sim.verbose to true\n"
          << "  --write-config=<path>  Write the merged config to an explicit file\n"
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

    void set_merged_config_output_path(
        const std::string& value,
        const std::string& flag_name,
        ProgramArgs& program_args) {
      if (value.empty()) {
        throw std::runtime_error("Missing value for " + flag_name);
      }
      program_args.merged_config_output_path = value;
    }

    void set_log_output_path(
        const std::string& value,
        const std::string& flag_name,
        ProgramArgs& program_args) {
      if (value.empty()) {
        throw std::runtime_error("Missing value for " + flag_name);
      }
      program_args.log_output_path = value;
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

      if (flag == "--validate-config") {
        program_args.validate_config_only = true;
        return;
      }

      if (flag == "--setup-only") {
        program_args.setup_only = true;
        return;
      }

      if (flag == "--verbose") {
        program_args.verbose_output = true;
        return;
      }

      if (flag.rfind("--output=", 0) == 0) {
        auto value = flag_value_from_equals_form(flag, "--output");
        if (value.empty()) {
          throw std::runtime_error("Missing value for --output");
        }
        program_args.output_path = value;
        return;
      }

      if (flag.rfind("--log=", 0) == 0) {
        set_log_output_path(flag_value_from_equals_form(flag, "--log"), "--log", program_args);
        return;
      }

      if (flag.rfind("--name=", 0) == 0) {
        auto value = flag_value_from_equals_form(flag, "--name");
        if (value.empty()) {
          throw std::runtime_error("Missing value for --name");
        }
        program_args.simulation_name = value;
        return;
      }

      if (flag.rfind("--seed=", 0) == 0) {
        set_random_seed(flag_value_from_equals_form(flag, "--seed"), "--seed", program_args);
        return;
      }

      if (flag.rfind("--write-config=", 0) == 0) {
        set_merged_config_output_path(flag_value_from_equals_form(flag, "--write-config"), "--write-config", program_args);
        return;
      }

      if (flag.rfind("--spins=", 0) == 0) {
        set_initial_spin_filename(flag_value_from_equals_form(flag, "--spins"), "--spins", program_args);
        return;
      }

      if (flag.rfind("--temp-directory=", 0) == 0) {
        std::string temp_directory = flag_value_from_equals_form(flag, "--temp-directory");
        if (temp_directory.empty()) {
          throw std::runtime_error("Missing value for --temp-directory");
        }
        program_args.temp_directory_path = temp_directory;
        return;
      }

      if (flag.rfind("--temp-dir=", 0) == 0) {
        std::string temp_directory = flag_value_from_equals_form(flag, "--temp-dir");
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

    std::string find_log_output_path_arg(int argc, char **argv) {
      ProgramArgs program_args;

      for (int n = 1; n < argc; ++n) {
        std::string arg = trim(argv[n]);

        if (flag_matches(arg, "--log")) {
          set_log_output_path(consume_flag_value(argc, argv, n, arg, "--log"), "--log", program_args);
        }
      }

      return program_args.log_output_path;
    }

    ProgramArgs parse_args(int argc, char **argv) {
      if (argc == 1) throw std::runtime_error("No config specified");

      ProgramArgs program_args;
      for (int n = 1; n < argc; ++n) {
        std::string arg = trim(argv[n]);

        if (flag_matches(arg, "--config")) {
          std::string config_value = consume_flag_value(argc, argv, n, arg, "--config");
          if (arg == "--config") {
            config_value = parse_config_value(argc, argv, n);
          }
          add_config_input(config_value, true, program_args);
          continue;
        }

        if (flag_matches(arg, "--temp-directory")) {
          program_args.temp_directory_path = consume_flag_value(argc, argv, n, arg, "--temp-directory");
          continue;
        }

        if (flag_matches(arg, "--temp-dir")) {
          program_args.temp_directory_path = consume_flag_value(argc, argv, n, arg, "--temp-dir");
          continue;
        }

        if (flag_matches(arg, "--log")) {
          set_log_output_path(consume_flag_value(argc, argv, n, arg, "--log"), "--log", program_args);
          continue;
        }

        if (flag_matches(arg, "--spins")) {
          set_initial_spin_filename(consume_flag_value(argc, argv, n, arg, "--spins"), "--spins", program_args);
          continue;
        }

        if (flag_matches(arg, "--seed")) {
          set_random_seed(consume_flag_value(argc, argv, n, arg, "--seed"), "--seed", program_args);
          continue;
        }

        if (flag_matches(arg, "--write-config")) {
          set_merged_config_output_path(consume_flag_value(argc, argv, n, arg, "--write-config"), "--write-config", program_args);
          continue;
        }

        if (flag_matches(arg, "--output")) {
          program_args.output_path = consume_flag_value(argc, argv, n, arg, "--output");
          continue;
        }

        if (flag_matches(arg, "--name")) {
          program_args.simulation_name = consume_flag_value(argc, argv, n, arg, "--name");
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
