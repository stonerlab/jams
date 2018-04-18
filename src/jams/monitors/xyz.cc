// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>

#include "jams/helpers/error.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"

#include "xyz.h"

#include "jblib/containers/array.h"
#include "jblib/math/equalities.h"


XyzMonitor::XyzMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  using namespace jblib;

  output_step_freq_ = settings["output_steps"];

  // settings for only outputting a slice
  if (settings.exists("slice_origin") ^ settings.exists("slice_size")) {
    jams_error("Xyz monitor requires both slice_origin and slice_size to be specificed;");
  }

  slice_spins.resize(0);

  if (settings.exists("slice_origin")) {
    for (int i = 0; i < 3; ++i) {
      slice_origin[i] = settings["slice_origin"][i];
    }
    for (int i = 0; i < 3; ++i) {
      slice_size[i] = settings["slice_size"][i];
    }
    // check which spins are inside the slice
    for (int i = 0; i < num_spins; ++i) {
      Vec3 pos = ::lattice->atom_position(i);

      // check if the current spin in inside the slice
      if (floats_are_greater_than_or_equal(pos[0], slice_origin[0]) && floats_are_less_than_or_equal(pos[0], slice_origin[0] + slice_size[0])
      &&  floats_are_greater_than_or_equal(pos[1], slice_origin[1]) && floats_are_less_than_or_equal(pos[1], slice_origin[1] + slice_size[1])
      &&  floats_are_greater_than_or_equal(pos[2], slice_origin[2]) && floats_are_less_than_or_equal(pos[2], slice_origin[2] + slice_size[2])) {
        slice_spins.push_back(i);
      }
    }
  }
}

void XyzMonitor::update(Solver * solver) {
  using namespace globals;

  if (solver->iteration()%output_step_freq_ == 0) {
    int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

    std::ofstream xyz_state_file(std::string(seedname+"_"+zero_pad_number(outcount)+"[0]yz").c_str());

    // file header
    xyz_state_file << "#";
    xyz_state_file << std::setw(9) << "spin num";
    xyz_state_file << std::setw(16) << "rx";
    xyz_state_file << std::setw(16) << "ry";
    xyz_state_file << std::setw(16) << "rz";
    xyz_state_file << std::setw(16) << "sx";
    xyz_state_file << std::setw(16) << "sy";
    xyz_state_file << std::setw(16) << "sz" << std::endl;

    if (slice_spins.size() > 0) {
      for (int i = 0, iend = slice_spins.size(); i < iend; ++i) {
        const int n = slice_spins[i];
        xyz_state_file << std::setw(9) << n;
        xyz_state_file << std::setw(16) << ::lattice->atom_position(n)[0] << std::setw(16) << ::lattice->atom_position(n)[1] << std::setw(16) << ::lattice->atom_position(n)[2];
        xyz_state_file << std::setw(16) << s(n,0) << std::setw(16) << s(n,1) << std::setw(16) <<  s(n, 2) << "\n";
      }
    } else {
      for (int n = 0; n < num_spins; ++n) {
        xyz_state_file << std::setw(9) << n;
        xyz_state_file << std::setw(16) << ::lattice->atom_position(n)[0] << std::setw(16) << ::lattice->atom_position(n)[1] << std::setw(16) << ::lattice->atom_position(n)[2];
        xyz_state_file << std::setw(16) << s(n,0) << std::setw(16) << s(n,1) << std::setw(16) <<  s(n, 2) << "\n";
      }
    }
    xyz_state_file.close();
  }
}

XyzMonitor::~XyzMonitor() {
}
