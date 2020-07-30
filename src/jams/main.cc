#include <cstdlib>
#include <iostream>
#include <version.h>
#include "jams/core/jams++.h"
#include "jams/core/args.h"
#include "jams/helpers/output.h"

int main(int argc, char **argv) {
  jams::output::initialise();
  auto program_args = jams::parse_args(argc, argv);

  if (program_args.version_only) {
    std::cout << "jams-" << semantic_version(jams::build::description) << std::endl;
    return EXIT_SUCCESS;
  }

  jams::initialize_simulation(program_args);
  if (!program_args.setup_only) {
    jams::run_simulation();
  }
  jams::cleanup_simulation();

  return EXIT_SUCCESS;
}