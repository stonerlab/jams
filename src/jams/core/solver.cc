// Copyright 2014 Joseph Barker. All rights reserved.

#include <algorithm>
#include <string>
#include <type_traits>
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
#include "jams/solvers/cpu_gse_rk4.h"
#include "jams/solvers/cpu_ll_lorentzian_rk4.h"
#include "jams/solvers/cpu_llg_additional_solvers.h"
#include "jams/solvers/cuda_llg_dm.h"
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
#include "jams/solvers/cuda_llg_rkmk4.h"
#include "jams/solvers/solver_descriptor.h"


void Solver::compute_fields() {
  if (hamiltonians_.empty()) return;

  for (auto& hh : hamiltonians_) {
    hh->calculate_fields(this->time());
  }

  std::copy(hamiltonians_[0]->ptr_field(), hamiltonians_[0]->ptr_field()+globals::num_spins3, globals::h.data());

  if (hamiltonians_.size() == 1) return;

  if constexpr (std::is_same_v<jams::Real, double>)
  {
    for (auto i = 1; i < hamiltonians_.size(); ++i) {
      cblas_daxpy(globals::num_spins3, 1.0, 
                      reinterpret_cast<const double*>(hamiltonians_[i]->ptr_field()), 1, 
                      reinterpret_cast<double*>(globals::h.data()), 1);
        }
      } else
      {
        for (auto i = 1; i < hamiltonians_.size(); ++i) {
          cblas_saxpy(globals::num_spins3, 1.0f, 
                      reinterpret_cast<const float*>(hamiltonians_[i]->ptr_field()), 1, 
                      reinterpret_cast<float*>(globals::h.data()), 1);
        }
      }

}


Solver* Solver::create(const libconfig::Setting &settings) {
  const auto descriptor = jams::solvers::describe_solver_setting(settings, *globals::config);
  const auto canonical_name = jams::solvers::canonical_solver_name(descriptor);
  std::cout << canonical_name << " solver\n";

  using jams::solvers::Backend;
  using jams::solvers::EquationKind;
  using jams::solvers::IntegratorKind;

  if (descriptor.backend == Backend::CPU) {
    if (descriptor.integrator == IntegratorKind::Null) {
      return new NullSolver(settings);
    }
    if (descriptor.integrator == IntegratorKind::Rotations) {
      return new RotationSolver(settings);
    }
    if (descriptor.integrator == IntegratorKind::MonteCarloMetropolis) {
      return new MetropolisMCSolver(settings);
    }
    if (descriptor.integrator == IntegratorKind::MonteCarloConstrained) {
      return new ConstrainedMCSolver(settings);
    }
    if (descriptor.integrator == IntegratorKind::MonteCarloMetadynamics) {
      return new MetadynamicsMetropolisSolver(settings);
    }
    if (descriptor.equation == EquationKind::LLG) {
      if (descriptor.integrator == IntegratorKind::Heun) {
        return new HeunLLGSolver(settings);
      }
      if (descriptor.integrator == IntegratorKind::RK4) {
        return new CPULLGRK4Solver(settings);
      }
      if (descriptor.integrator == IntegratorKind::RKMK2) {
        return new CPULLGRKMK2Solver(settings);
      }
      if (descriptor.integrator == IntegratorKind::RKMK4) {
        return new CPULLGRKMK4Solver(settings);
      }
      if (descriptor.integrator == IntegratorKind::SemiImplicit) {
        return new CPULLGSemiImplicitSolver(settings);
      }
      if (descriptor.integrator == IntegratorKind::DM) {
        return new CPULLGDMSolver(settings);
      }
    }
    if (descriptor.equation == EquationKind::GSE && descriptor.integrator == IntegratorKind::RK4) {
      return new CPUGSERK4Solver(settings);
    }
    if (descriptor.equation == EquationKind::LLLorentzian && descriptor.integrator == IntegratorKind::RK4) {
      return new CPULLLorentzianRK4Solver(settings);
    }
  }

  if (descriptor.backend == Backend::GPU) {
#if HAS_CUDA
    if (descriptor.equation == EquationKind::LLG) {
      if (descriptor.integrator == IntegratorKind::Heun) {
        return new CUDAHeunLLGSolver(settings);
      }
      if (descriptor.integrator == IntegratorKind::RK4) {
        return new CUDALLGRK4Solver(settings);
      }
      if (descriptor.integrator == IntegratorKind::RKMK2) {
        return new CUDALLGRKMK2Solver(settings);
      }
      if (descriptor.integrator == IntegratorKind::RKMK4) {
        return new CUDALLGRKMK4Solver(settings);
      }
      if (descriptor.integrator == IntegratorKind::SemiImplicit) {
        return new CUDALLGSemiImplictSolver(settings);
      }
      if (descriptor.integrator == IntegratorKind::DM) {
        return new CUDALLGDMSolver(settings);
      }
    }
    if (descriptor.equation == EquationKind::GSE && descriptor.integrator == IntegratorKind::RK4) {
      return new CUDAGSERK4Solver(settings);
    }
    if (descriptor.equation == EquationKind::LLLorentzian && descriptor.integrator == IntegratorKind::RK4) {
      return new CUDALLLorentzianRK4Solver(settings);
    }
#else
    throw std::runtime_error("CUDA solver requested but JAMS was built without CUDA support");
#endif
  }

  throw std::runtime_error("unsupported solver configuration " + canonical_name);
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
