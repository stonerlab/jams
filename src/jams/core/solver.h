// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SOLVER_H
#define JAMS_CORE_SOLVER_H

#include <memory>
#include <vector>
#include <libconfig.h++>

#include "jams/core/thermostat.h"
#include "jams/core/physics.h"
#include "jams/core/monitor.h"
#include "jams/core/hamiltonian.h"

class Solver {
 public:
  Solver() = default;
  ~Solver() = default;

  virtual void initialize(const libconfig::Setting& settings) = 0;
  virtual void run() = 0;

  virtual std::string name() const = 0;

  virtual bool is_cuda_solver() const { return false; }

  bool is_converged();
  virtual bool is_running();

  inline int iteration() const {
    return iteration_;
  }

  inline double time() const {
    return iteration_ * step_size_;
  }

  inline double time_step() const {
    return step_size_;
  }

  inline int max_steps() const {
    return max_steps_;
  }

  inline const Physics * physics() const {
    return physics_module_.get();
  }

  inline Thermostat * thermostat() const {
    return thermostat_.get();
  }

  void register_physics_module(Physics* package);
  void update_physics_module();

  void register_thermostat(Thermostat* thermostat);
  void update_thermostat();

  void register_monitor(Monitor* monitor);
  void register_hamiltonian(Hamiltonian* hamiltonian);

  virtual void notify_monitors();

  virtual void compute_fields();

  std::vector<std::unique_ptr<Hamiltonian>>& hamiltonians() {
    return hamiltonians_;
  }

  std::vector<std::unique_ptr<Monitor>>& monitors() {
    return monitors_;
  }

  static Solver* create(const libconfig::Setting &setting);
 protected:
    int iteration_ = 0;
    int max_steps_ = 0;
    int min_steps_ = 0;

    double step_size_ = 1.0;

  std::unique_ptr<Physics> physics_module_;
  std::unique_ptr<Thermostat> thermostat_;
  std::vector<std::unique_ptr<Monitor>> monitors_;
  std::vector<std::unique_ptr<Hamiltonian>> hamiltonians_;
};

#endif  // JAMS_CORE_SOLVER_H
