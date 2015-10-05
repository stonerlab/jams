// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SOLVER_H
#define JAMS_CORE_SOLVER_H

#include <cassert>
#include <string>
#include <vector>

#include <fftw3.h>

// forward declarations
class Physics;
class Monitor;
class Hamiltonian;

class Solver {
 public:
  Solver()
  : initialized_(false),
    is_cuda_solver_(false),
    iteration_(0),
    time_step_(0.0),
    real_time_step_(0.0),
    physics_module_()
  {}

  virtual ~Solver();

  virtual void initialize(int argc, char **argv, double dt) = 0;
  virtual void run() = 0;

  bool is_converged();

  inline bool is_cuda_solver() const {
    return is_cuda_solver_;
  }

  inline int iteration() const {
    return iteration_;
  }

  inline double time() const {
    return iteration_*real_time_step_;
  }

  inline const Physics * physics() const {
    return physics_module_;
  }

  void register_physics_module(Physics* package);
  void update_physics_module();

  void register_monitor(Monitor* monitor);
  void register_hamiltonian(Hamiltonian* hamiltonian);

  virtual void notify_monitors();

  void compute_fields();
  void compute_energy();

  virtual inline double * dev_ptr_spin() {
    assert(is_cuda_solver_);
    return NULL;
  }

  std::vector<Hamiltonian*>& hamiltonians() {
    return hamiltonians_;
  }

  static Solver* create(const std::string &solver_name);
 protected:
  bool initialized_;
  bool is_cuda_solver_;

  int    iteration_;
  double time_step_;
  double real_time_step_;

  Physics*                  physics_module_;
  std::vector<Monitor*>     monitors_;
  std::vector<Hamiltonian*> hamiltonians_;

  fftw_plan spin_fft_forward_transform;
  fftw_plan field_fft_backward_transform;
  fftw_plan interaction_fft_transform;
};

#endif  // JAMS_CORE_SOLVER_H
