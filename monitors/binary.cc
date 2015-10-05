// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>
#include <iostream>
#include <fstream>


#include "core/globals.h"
#include "core/lattice.h"
#include "core/utils.h"

#include "monitors/binary.h"

#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"
#include "jblib/math/equalities.h"

BinaryMonitor::BinaryMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  using namespace jblib;

  ::output.write("\nInitialising binary monitor...\n");

  is_equilibration_monitor_ = false;

  is_file_overwrite_mode = false;
  settings.lookupValue("overwrite", is_file_overwrite_mode);

  output_step_freq_ = settings["output_steps"];
}

void BinaryMonitor::update(Solver * solver) {
  using namespace globals;
  if (solver->iteration()%output_step_freq_ == 0) {
    int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

    std::ofstream bin_file;

    if (is_file_overwrite_mode) {
      bin_file.open(std::string(seedname+".bin").c_str(), std::ios::binary | std::ios::trunc);
    } else {
      bin_file.open(std::string(seedname+"_"+zero_pad_number(outcount)+".bin").c_str(), std::ios::binary);
    }

    // pointers must be reinterpreted as a char *
    bin_file.write(reinterpret_cast<char*>(&num_spins), sizeof(int));
    bin_file.write(reinterpret_cast<char*>(s.data()), sizeof(double)*num_spins);
    bin_file.close();
  }
}

BinaryMonitor::~BinaryMonitor() {
}
