#include <cstdlib>

#include "jams/core/jams++.h"

int main(int argc, char **argv) {
  jams_initialize(argc, argv);
  jams_run();
  jams_finish();
  return EXIT_SUCCESS;
}