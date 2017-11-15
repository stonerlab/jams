// Copyright 2014 Joseph Barker. All rights reserved.

#include <algorithm>
#include <string>
#include <jams/core/config.h>

#include "jams/core/error.h"
#include "jams/core/physics.h"
#include "jblib/containers/array.h"
#include "jams/core/blas.h"
#include "jams/core/solver.h"
#include "jams/core/hamiltonian.h"
#include "jams/core/monitor.h"
#include "jams/core/consts.h"

#include "jams/core/utils.h"
#include "jams/core/globals.h"
#include "jams/core/defaults.h"

#include "jams/solvers/cuda_heunllg.h"
#include "jams/solvers/heunllg.h"
#include "jams/solvers/metropolismc.h"
#include "jams/solvers/constrainedmc.h"
#include "jams/solvers/cuda_constrainedmc.h"

Solver::~Solver() {
  for (auto& m : monitors_) {
    if (m) {
      delete m;
      m = nullptr;
    }
  }
}


void Solver::initialize(const libconfig::Setting& settings) {
  assert(!initialized_);

  initialized_ = true;
}


void Solver::run() {
}


void Solver::compute_fields() {
  globals::h.zero();

  for (auto& hh : hamiltonians_) {
    hh->calculate_fields();
  }

  // sum hamiltonian field contributions into effective field
  for (auto& hh : hamiltonians_) {
    cblas_daxpy(globals::num_spins3, 1.0, hh->ptr_field(), 1, globals::h.data(), 1);
  }
}


Solver* Solver::create(const libconfig::Setting &settings) {
  std::string module_name = jams::default_physics_module;
  settings.lookupValue("module", module_name);
  module_name = lowercase(module_name);

  if (module_name == "llg-heun-cpu") {
    return new HeunLLGSolver;
  }

  if (module_name == "monte-carlo-metropolis-cpu") {
    return new MetropolisMCSolver;
  }

  if (module_name == "monte-carlo-constrained-cpu") {
    return new ConstrainedMCSolver;
  }
#ifdef CUDA
  if (module_name == "llg-heun-gpu") {
    return new CUDAHeunLLGSolver;
  }

  if (module_name == "monte-carlo-constrained-gpu") {
    return new CudaConstrainedMCSolver;
  }
#endif

  jams_error("Unknown solver '%s' selected.", module_name.c_str());
  return nullptr;
}


void Solver::register_physics_module(Physics* package) {
    physics_module_ = package;
}


void Solver::update_physics_module() {
    physics_module_->update(iteration_, time(), time_step_);
}


void Solver::register_monitor(Monitor* monitor) {
  monitors_.push_back(monitor);
}


void Solver::register_hamiltonian(Hamiltonian* hamiltonian) {
  hamiltonians_.push_back(hamiltonian);
}


void Solver::notify_monitors() {
  for (auto& m : monitors_) {
    if (m->is_updating(iteration_)) {
      m->update(this);
    }
  }
}

bool Solver::is_running() {
  return iteration_ < max_steps_;
}

bool Solver::is_converged() {
  if (iteration_ < min_steps_) {
    return false;
  }

  for (auto& m : monitors_) {
    if (m->is_converged()) {
      return true;
    }
  }
  return false;
}
