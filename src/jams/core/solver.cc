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
#include "jams/core/thermostat.h"

#include "jams/core/base.h"
#include "jams/helpers/utils.h"
#include "jams/core/globals.h"
#include "jams/helpers/defaults.h"

#include "jams/solvers/null_solver.h"
#include "jams/solvers/cuda_llg_heun.h"
#include "jams/solvers/cuda_llg_rk4.h"
#include "jams/solvers/cuda_metadynamics_llg_rk4.h"
#include "jams/solvers/cuda_ll_lorentzian_rk4.h"
#include "jams/solvers/cpu_llg_heun.h"
#include "jams/solvers/cpu_rotations.h"
#include "jams/solvers/cpu_monte_carlo_metropolis.h"
#include "jams/solvers/cpu_monte_carlo_constrained.h"
#include "jams/solvers/cpu_metadynamics_metropolis_solver.h"


#define DEFINED_SOLVER(name, type) \
{ \
if (lowercase(settings["module"]) == name) { \
std::cout << name << " solver \n"; \
return new type; \
} \
}

using namespace std;


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
  DEFINED_SOLVER("null", NullSolver);
  DEFINED_SOLVER("rotations-cpu", RotationSolver);
  DEFINED_SOLVER("llg-heun-cpu", HeunLLGSolver);
  DEFINED_SOLVER("monte-carlo-metropolis-cpu", MetropolisMCSolver);
  DEFINED_SOLVER("monte-carlo-constrained-cpu", ConstrainedMCSolver);
  DEFINED_SOLVER("monte-carlo-metadynamics-cpu", MetadynamicsMetropolisSolver);

#if HAS_CUDA
  DEFINED_SOLVER("llg-heun-gpu", CUDAHeunLLGSolver);
  DEFINED_SOLVER("llg-rk4-gpu", CUDALLGRK4Solver);
  DEFINED_SOLVER("llg-metadynamics-rk4-gpu", CUDAMetadynamicsLLGRK4Solver);
  DEFINED_SOLVER("ll-lorentzian-rk4-gpu", CUDALLLorentzianRK4Solver);
#endif

  throw std::runtime_error("unknown solver " + std::string(settings["module"].c_str()));
}


void Solver::register_physics_module(Physics* package) {
    physics_module_.reset(package);
}


void Solver::update_physics_module() {
    physics_module_->update(iteration_, time(), step_size_);
}

void Solver::register_thermostat(Thermostat* thermostat) {
  thermostat_.reset(thermostat);
}


void Solver::update_thermostat() {
  thermostat_->set_temperature(physics_module_->temperature());
  thermostat_->update();
}


void Solver::register_monitor(Monitor* monitor) {
  monitors_.push_back(static_cast<unique_ptr<Monitor>>(monitor));
}


void Solver::register_hamiltonian(Hamiltonian* hamiltonian) {
  hamiltonians_.push_back(static_cast<unique_ptr<Hamiltonian>>(hamiltonian));
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

Monitor::ConvergenceStatus Solver::convergence_status() {
  if (iteration_ < min_steps_) {
    return Monitor::ConvergenceStatus::kNotConverged;
  }

  bool all_monitors_disabled = true;
  for (auto& m : monitors_) {
    const auto& status = m->convergence_status();

    all_monitors_disabled &= (status == Monitor::ConvergenceStatus::kDisabled);

    if (status == Monitor::ConvergenceStatus::kNotConverged) {
      return status;
    }
  }

  if (all_monitors_disabled) {
    return Monitor::ConvergenceStatus::kDisabled;
  }

  return Monitor::ConvergenceStatus::kConverged;
}

#undef DEFINED_SOLVER