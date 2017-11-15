// Copyright 2014 Joseph Barker. All rights reserved.

#include <algorithm>
#include <string>

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


void Solver::initialize(int argc, char **argv, double idt) {
  assert(!initialized_);
  real_time_step_ = idt;
  time_step_ = idt * kGyromagneticRatio;
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


Solver* Solver::create(const std::string &solver_name) {
  if (capitalize(solver_name) == "LLG-HEUN-CPU" || capitalize(solver_name) == "HEUNLLG") {
    return new HeunLLGSolver;
  }

  if (capitalize(solver_name) == "MONTE-CARLO-METROPOLIS-CPU" || capitalize(solver_name) == "METROPOLISMC") {
    return new MetropolisMCSolver;
  }

  if (capitalize(solver_name) == "MONTE-CARLO-CONSTRAINED-CPU" || capitalize(solver_name) == "CONSTRAINEDMC") {
    return new ConstrainedMCSolver;
  }
#ifdef CUDA
  if (capitalize(solver_name) == "LLG-HEUN-GPU" || capitalize(solver_name) == "CUDAHEUNLLG") {
    return new CUDAHeunLLGSolver;
  }

  if (capitalize(solver_name) == "MONTE-CARLO-CONSTRAINED-GPU" || capitalize(solver_name) == "CUDACONSTRAINEDMC") {
    return new CudaConstrainedMCSolver;
  }
#endif

  jams_error("Unknown solver '%s' selected.", solver_name.c_str());
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


bool Solver::is_converged() {
  for (auto& m : monitors_) {
    if (m->is_converged()) {
      return true;
    }
  }
  return false;
}
