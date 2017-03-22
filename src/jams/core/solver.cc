// Copyright 2014 Joseph Barker. All rights reserved.

#include <algorithm>
#include <string>

#include "jams/core/error.h"
#include "jams/core/physics.h"
#include "jams/core/types.h"
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

#ifdef MKL
#include <mkl_spblas.h>
#endif

Solver::~Solver() {
  for (int i = 0, iend = monitors_.size(); i < iend; ++i) {
    delete monitors_[i];
  }
}

//---------------------------------------------------------------------

void Solver::initialize(int argc, char **argv, double idt) {
  using namespace globals;
  if (initialized_ == true) {
    jams_error("Solver is already initialized");
  }

  real_time_step_ = idt;
  time_step_ = idt*kGyromagneticRatio;

  initialized_ = true;
}

//---------------------------------------------------------------------

void Solver::run() {
}

//---------------------------------------------------------------------

void Solver::compute_fields() {
  using namespace globals;

  // zero the effective field array
  std::fill(h.data(), h.data()+num_spins3, 0.0);

  // calculate each hamiltonian term's fields
  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    (*it)->calculate_fields();
  }

  // sum hamiltonian field contributions into effective field
  for (std::vector<Hamiltonian*>::iterator it = hamiltonians_.begin() ; it != hamiltonians_.end(); ++it) {
    cblas_daxpy(num_spins3, 1.0, (*it)->ptr_field(), 1, h.data(), 1);
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
  return NULL;
}

//---------------------------------------------------------------------

void Solver::register_physics_module(Physics* package) {
    physics_module_ = package;
}

//---------------------------------------------------------------------

void Solver::update_physics_module() {
    physics_module_->update(iteration_, time(), time_step_);
}

//---------------------------------------------------------------------

void Solver::register_monitor(Monitor* monitor) {
  monitors_.push_back(monitor);
}

//---------------------------------------------------------------------

void Solver::register_hamiltonian(Hamiltonian* hamiltonian) {
  hamiltonians_.push_back(hamiltonian);
}

//---------------------------------------------------------------------

void Solver::notify_monitors() {
  for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
    if((*it)->is_updating(iteration_)){
      (*it)->update(this);
    }
  }
}

bool Solver::is_converged() {
  for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
    if((*it)->is_converged()){
      return true;
    }
  }
  return false;
}
