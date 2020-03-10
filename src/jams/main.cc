#include <cstdlib>
#include <iostream>
#include <version.h>
#include "jams/core/jams++.h"
#include "jams/core/args.h"
#include "jams/helpers/output.h"
#include "jams/helpers/filesystem.h"
#include "jams/common.h"

int main(int argc, char **argv) {
  jams::output::initialise();
  std::cout << "\nJAMS++ " << jams::build::version << std::endl;

  auto program_args = jams::parse_args(argc, argv);

  if (!program_args.output_path.empty()) {
    jams::instance().set_output_dir(program_args.output_path);
  }

  jams::initialize_simulation(program_args);

  if (!program_args.setup_only) {
    jams::run_simulation();
  }
  jams::cleanup_simulation();

  return EXIT_SUCCESS;
}