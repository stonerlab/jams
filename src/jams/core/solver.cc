// Copyright 2014 Joseph Barker. All rights reserved.

#include <algorithm>
#include <string>
#include <jams/interface/config.h>

#include "jams/helpers/error.h"
#include "jams/core/physics.h"
#include "jams/interface/blas.h"
#include "jams/core/solver.h"
#include "hamiltonian.h"
#include "jams/core/monitor.h"
#include "jams/helpers/consts.h"

#include "jams/core/base.h"
#include "jams/helpers/utils.h"
#include "jams/core/globals.h"
#include "jams/helpers/defaults.h"

#include "jams/solvers/cuda_llg_heun.h"
#include "jams/solvers/cpu_llg_heun.h"
#include "jams/solvers/cpu_rotations.h"
#include "jams/solvers/cpu_monte_carlo_metropolis.h"
#include "jams/solvers/cpu_monte_carlo_constrained.h"
#include "jams/solvers/cpu_metadynamics_metropolis_solver.h"


using namespace std;

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
  set_name(jams::config_required<string>(settings, "module"));
  set_verbose(jams::config_optional<bool>(settings, "verbose", false));
  set_debug(jams::config_optional<bool>(settings, "debug", false));

  cout << "  " << name() << " solver\n";

  initialized_ = true;
}


void Solver::run() {
}


void Solver::compute_fields() {
  if (hamiltonians_.empty()) return;

  for (auto& hh : hamiltonians_) {
    hh->calculate_fields();
  }

  std::copy(hamiltonians_[0]->ptr_field(), hamiltonians_[0]->ptr_field()+globals::num_spins3, globals::h.data());

  if (hamiltonians_.size() == 1) return;

  for (auto i = 1; i < hamiltonians_.size(); ++i) {
    cblas_daxpy(globals::num_spins3, 1.0, hamiltonians_[i]->ptr_field(), 1, globals::h.data(), 1);
  }
}


Solver* Solver::create(const libconfig::Setting &settings) {
  auto module_name = jams::config_required<string>(settings, "module");
  module_name = lowercase(module_name);

  if (module_name == "rotations-cpu") {
    return new RotationSolver;
  }

  if (module_name == "llg-heun-cpu") {
    return new HeunLLGSolver;
  }

  if (module_name == "monte-carlo-metropolis-cpu") {
    return new MetropolisMCSolver;
  }

  if (module_name == "monte-carlo-metadynamics-cpu") {
    return new MetadynamicsMetropolisSolver;
  }

  if (module_name == "monte-carlo-constrained-cpu") {
    return new ConstrainedMCSolver;
  }
#if HAS_CUDA
  if (module_name == "llg-heun-gpu") {
    return new CUDAHeunLLGSolver;
  }
#endif

  throw std::runtime_error("unknown solver " + module_name);
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
