// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/utils.h"

#include "monitors/hdf5.h"

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

Hdf5Monitor::Hdf5Monitor(const libconfig::Setting &settings)
: Monitor(settings) {
    using namespace globals;

    ::output.write("\nInitialising HDF5 monitor...\n");

    is_equilibration_monitor_ = false;
    output_step_freq_ = settings["output_steps"];
}

void Hdf5Monitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

  if (iteration%output_step_freq_ == 0) {
  }
}

Hdf5Monitor::~Hdf5Monitor() {
}
