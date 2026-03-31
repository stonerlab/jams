#include <cstdlib>
#include <iostream>
#include <version.h>
#include "jams/core/jams++.h"
#include "jams/core/args.h"
#include "jams/helpers/output.h"

int main(int argc, char **argv) {
  try {
    const auto log_output_path = jams::find_log_output_path_arg(argc, argv);
    if (!log_output_path.empty()) {
      jams::output::redirect_standard_streams(log_output_path);
    }

    jams::output::initialise();
    auto program_args = jams::parse_args(argc, argv);

    if (program_args.help_only) {
      jams::print_usage(std::cout);
      return EXIT_SUCCESS;
    }

    if (program_args.version_only) {
      std::cout << "jams-" << semantic_version(jams::build::description) << std::endl;
      return EXIT_SUCCESS;
    }

    if (program_args.validate_config_only) {
      jams::validate_config(program_args);
      return EXIT_SUCCESS;
    }

    jams::initialize_simulation(program_args);
    if (!program_args.setup_only) {
      jams::run_simulation();
    }
    jams::cleanup_simulation();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n";
    jams::print_usage(std::cerr);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
