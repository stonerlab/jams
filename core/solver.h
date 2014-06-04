// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_SOLVER_H
#define JAMS_CORE_SOLVER_H

#include <fftw3.h>
#include "core/globals.h"
#include "core/physics.h"
#include "core/monitor.h"
#include "jblib/containers/vec.h"

class Solver {
 public:
  Solver()
  : initialized_(false),
    iteration_(0),
    time_step_(0.0),
    real_time_step_(0.0),
    physics_module_()
  {}

  virtual ~Solver() {
    for (int i = 0, iend = monitors_.size(); i < iend; ++i) {
      delete monitors_[i];
    }
  }

  virtual void initialize(int argc, char **argv, double dt) = 0;
  virtual void run() = 0;

  inline double time() const {
    return iteration_*real_time_step_;
  }

  inline void register_physics_module(Physics* package) {
    physics_module_ = package;
  }

  inline void update_physics_module() {
    physics_module_->update(iteration_, time(), time_step_);
  }

  inline void register_monitor(Monitor* monitor) {
    monitors_.push_back(monitor);
  }

  virtual inline void notify_monitors() {
    for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
      if((*it)->is_updating(iteration_)){
        (*it)->update(iteration_, time(), physics_module_->temperature(), physics_module_->applied_field());
      }
    }
  }

  void compute_fields();
  void compute_energy();

  static Solver* create(const std::string &solver_name);
 protected:
  bool initialized_;

  int    iteration_;
  double time_step_;
  double real_time_step_;

  Physics*              physics_module_;
  std::vector<Monitor*> monitors_;

  fftw_plan spin_fft_forward_transform;
  fftw_plan field_fft_backward_transform;
  fftw_plan interaction_fft_transform;
};

#endif  // JAMS_CORE_SOLVER_H
