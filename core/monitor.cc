// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/monitor.h"

#include "core/globals.h"
#include "core/solver.h"

void Monitor::initialize() {
  if (initialized == true) {
    jams_error("Monitor is already initialized");
  }

  output.write("Initialising monitor\n");
}

void Monitor::run() {
}

void Monitor::write(Solver *solver) {
}

void Monitor::initialize_convergence(ConvergenceType type, const double meanTol,
  const double devTol) {
}

bool Monitor::has_converged() {
  return false;
}
