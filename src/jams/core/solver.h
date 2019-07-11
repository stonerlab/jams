// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SOLVER_H
#define JAMS_CORE_SOLVER_H

#include <cstddef>
#include <cassert>
#include <iosfwd>
#include <vector>
#include <libconfig.h++>

#include "jams/interface/fft.h"
#include "jams/core/base.h"

// forward declarations
class Physics;
class Monitor;
class Hamiltonian;
class Thermostat;

class Solver : public Base {
 public:
  Solver() = default;

  virtual ~Solver();

  virtual void initialize(const libconfig::Setting& settings) = 0;
  virtual void run() = 0;

  bool is_converged();
  bool is_running();

  inline bool is_cuda_solver() const {
    return is_cuda_solver_;
  }

  inline int iteration() const {
    return iteration_;
  }

  inline double time() const {
    return iteration_ * time_step_;
  }

  inline double time_step() const {
    return time_step_;
  }

  inline int max_steps() const {
    return max_steps_;
  }

  inline const Physics * physics() const {
    return physics_module_;
  }

  inline Thermostat * thermostat() const {
    return thermostat_;
  }

  void register_physics_module(Physics* package);
  void update_physics_module();

  void register_monitor(Monitor* monitor);
  void register_hamiltonian(Hamiltonian* hamiltonian);

  virtual void notify_monitors();

  void compute_fields();

  std::vector<Hamiltonian*>& hamiltonians() {
    return hamiltonians_;
  }

  std::vector<Monitor*>& monitors() {
    return monitors_;
  }

  static Solver* create(const libconfig::Setting &setting);
 protected:
    bool initialized_ = false;
    bool is_cuda_solver_ = false;

    int iteration_ = 0;
    int max_steps_ = 0;
    int min_steps_ = 0;

    double time_step_ = 1.0;

  Physics*                  physics_module_ = nullptr;
  Thermostat*               thermostat_ = nullptr;
  std::vector<Monitor*>     monitors_;
  std::vector<Hamiltonian*> hamiltonians_;
};

#endif  // JAMS_CORE_SOLVER_H
