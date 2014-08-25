// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/utils.h"

#include "monitors/vtu.h"

VtuMonitor::VtuMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Vtu monitor...\n");

  is_equilibration_monitor_ = false;
  output_step_freq_ = settings["output_steps"];
}

void VtuMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

  if (iteration%output_step_freq_ == 0) {
    int outcount = iteration/output_step_freq_;  // int divisible by modulo above

    std::ofstream vtu_state_file
    (std::string(seedname+"_"+zero_pad_number(outcount)+".vtu").c_str());

    lattice.output_spin_state_as_vtu(vtu_state_file);

    vtu_state_file.close();
  }
}

VtuMonitor::~VtuMonitor() {
}
