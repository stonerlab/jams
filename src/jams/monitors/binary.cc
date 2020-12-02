// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <fstream>

#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/output.h"

#include "binary.h"

BinaryMonitor::BinaryMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;

  is_file_overwrite_mode = false;
  settings.lookupValue("overwrite", is_file_overwrite_mode);
}

void BinaryMonitor::update(Solver * solver) {
  using namespace globals;
  int outcount = solver->iteration()/output_step_freq_;

  std::ofstream bin_file;

  if (is_file_overwrite_mode) {
    bin_file.open(jams::output::full_path_filename(".bin"), std::ios::binary | std::ios::trunc);
  } else {
    bin_file.open(jams::output::full_path_filename_series(".bin", outcount), std::ios::binary);
  }

  // pointers must be reinterpreted as a char *
  bin_file.write(reinterpret_cast<char*>(&num_spins), sizeof(int));
  bin_file.write(reinterpret_cast<char*>(s.data()), sizeof(double)*num_spins);
  bin_file.close();
}