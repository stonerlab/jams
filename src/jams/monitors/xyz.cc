// Copyright 2014 Joseph Barker. All rights reserved.

#include <string>
#include <iomanip>
#include <fstream>

#include "jams/helpers/error.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/output.h"
#include "jams/interface/config.h"

#include "xyz.h"
#include <jams/helpers/exception.h>

XyzMonitor::XyzMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  output_step_freq_ = jams::config_required<int>(settings, "output_steps");

  // settings for only outputting a slice
  jams::require_settings_together(settings, {"slice_origin", "slice_size"});

  slice_spins.resize(0);

  if (settings.exists("slice_origin")) {
    slice_origin = jams::read_vec_setting<double, 3>(settings["slice_origin"], "slice_origin");
    slice_size = jams::read_vec_setting<double, 3>(settings["slice_size"], "slice_size");
    // check which spins are inside the slice
    for (int i = 0; i < globals::num_spins; ++i) {
      jams::Vec<double, 3> pos = globals::lattice->lattice_site_position_cart(i);

      // check if the current spin in inside the slice
      if (definately_greater_than(pos[0], slice_origin[0], jams::defaults::lattice_tolerance) && definately_less_than(pos[0], slice_origin[0] + slice_size[0], jams::defaults::lattice_tolerance)
          &&  definately_greater_than(pos[1], slice_origin[1], jams::defaults::lattice_tolerance) && definately_less_than(pos[1], slice_origin[1] + slice_size[1], jams::defaults::lattice_tolerance)
          &&  definately_greater_than(pos[2], slice_origin[2], jams::defaults::lattice_tolerance) && definately_less_than(pos[2], slice_origin[2] + slice_size[2], jams::defaults::lattice_tolerance)) {
        slice_spins.push_back(i);
      }
    }
  }
}

void XyzMonitor::update(Solver& solver) {
  if (solver.iteration()%output_step_freq_ == 0) {
    int outcount = solver.iteration()/output_step_freq_;  // int divisible by modulo above

    const auto spins = globals::s.host_view();
    std::ofstream xyz_state_file(jams::output::full_path_filename_series(".xyz", outcount));

    // file header
    xyz_state_file << "#";
    xyz_state_file << std::setw(9) << "spin num";
    xyz_state_file << std::setw(16) << "rx";
    xyz_state_file << std::setw(16) << "ry";
    xyz_state_file << std::setw(16) << "rz";
    xyz_state_file << std::setw(16) << "sx";
    xyz_state_file << std::setw(16) << "sy";
    xyz_state_file << std::setw(16) << "sz" << std::endl;

    if (!slice_spins.empty()) {
      for (const auto n : slice_spins) {
        xyz_state_file << std::setw(9) << n;
        xyz_state_file << std::setw(16) << globals::lattice->lattice_site_position_cart(n)[0] << std::setw(16) << globals::lattice->lattice_site_position_cart(
            n)[1] << std::setw(16) << globals::lattice->lattice_site_position_cart(
            n)[2];
        xyz_state_file << std::setw(16) << spins(n,0) << std::setw(16) << spins(n,1) << std::setw(16) <<  spins(n, 2) << "\n";
      }
    } else {
      for (int n = 0; n < globals::num_spins; ++n) {
        xyz_state_file << std::setw(9) << n;
        xyz_state_file << std::setw(16) << globals::lattice->lattice_site_position_cart(n)[0] << std::setw(16) << globals::lattice->lattice_site_position_cart(
            n)[1] << std::setw(16) << globals::lattice->lattice_site_position_cart(
            n)[2];
        xyz_state_file << std::setw(16) << spins(n,0) << std::setw(16) << spins(n,1) << std::setw(16) <<  spins(n, 2) << "\n";
      }
    }
    xyz_state_file.close();
  }
}
