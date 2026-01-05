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
#include "jams/solvers/cuda_gse_rk4.h"
#include "jams/solvers/cuda_llg_heun.h"
#include "jams/solvers/cuda_llg_rk4.h"
#include "jams/solvers/cuda_ll_lorentzian_rk4.h"
#include "jams/solvers/cuda_rk4_llg_sot.h"
#include "jams/solvers/cuda_llg_semi_implicit.h"
#include "jams/solvers/cpu_llg_heun.h"
#include "jams/solvers/cpu_rotations.h"
#include "jams/solvers/cpu_monte_carlo_metropolis.h"
#include "jams/solvers/cpu_monte_carlo_constrained.h"
#include "jams/solvers/cpu_metadynamics_metropolis_solver.h"
#include "jams/solvers/cuda_llg_rkmk2.h"


#define DEFINED_SOLVER(name, type, settings) \
{ \
if (lowercase(settings["module"]) == name) { \
std::cout << name << " solver \n"; \
return new type(settings); \
} \
}


void Solver::compute_fields() {
  if (hamiltonians_.empty()) return;

  for (auto& hh : hamiltonians_) {
    hh->calculate_fields(this->time());
  }

  std::copy(hamiltonians_[0]->ptr_field(), hamiltonians_[0]->ptr_field()+globals::num_spins3, globals::h.data());

  if (hamiltonians_.size() == 1) return;

  for (auto i = 1; i < hamiltonians_.size(); ++i) {
    cblas_daxpy(globals::num_spins3, 1.0, hamiltonians_[i]->ptr_field(), 1, globals::h.data(), 1);
  }
}


Solver* Solver::create(const libconfig::Setting &settings) {
  DEFINED_SOLVER("null", NullSolver, settings);
  DEFINED_SOLVER("rotations-cpu", RotationSolver, settings);
  DEFINED_SOLVER("llg-heun-cpu", HeunLLGSolver, settings);
  DEFINED_SOLVER("monte-carlo-metropolis-cpu", MetropolisMCSolver, settings);
  DEFINED_SOLVER("monte-carlo-constrained-cpu", ConstrainedMCSolver, settings);
  DEFINED_SOLVER("monte-carlo-metadynamics-cpu", MetadynamicsMetropolisSolver, settings);

#if HAS_CUDA
  DEFINED_SOLVER("gse-rk4-gpu", CUDAGSERK4Solver, settings);
  DEFINED_SOLVER("llg-heun-gpu", CUDAHeunLLGSolver, settings);
  DEFINED_SOLVER("llg-rk4-gpu", CUDALLGRK4Solver, settings);
  DEFINED_SOLVER("ll-lorentzian-rk4-gpu", CUDALLLorentzianRK4Solver, settings);
  DEFINED_SOLVER("llg-sot-rk4-gpu", CudaRK4LLGSOTSolver, settings);
  DEFINED_SOLVER("llg-simp-gpu", CUDALLGSemiImplictSolver, settings);
  DEFINED_SOLVER("llg-rkmk2-gpu", CUDALLGRKMK2Solver, settings);
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
  monitors_.push_back(static_cast<std::unique_ptr<Monitor>>(monitor));
}


void Solver::register_hamiltonian(Hamiltonian* hamiltonian) {
  hamiltonians_.push_back(static_cast<std::unique_ptr<Hamiltonian>>(hamiltonian));
}


void Solver::notify_monitors() {
  for (auto& m : monitors_) {
    if (m->is_updating(iteration_)) {
      m->update(*this);
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