// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDASOLVER_H
#define JAMS_CORE_CUDASOLVER_H

#include "core/cuda_defs.h"
#include "core/globals.h"
#include "core/solver.h"
#include "core/physics.h"
#include "core/monitor.h"
#include "jblib/containers/cuda_array.h"
#include "jblib/containers/vec.h"

class CudaSolver : public Solver {
 public:
  CudaSolver() {};
  ~CudaSolver();

  void initialize(int argc, char **argv, double dt);
  void run();

  inline void notify_monitors() {
    for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
      (*it)->update(iteration_, time(), physics_module_->temperature(), physics_module_->applied_field());
    }
  }

  void compute_fields();
  void compute_energy();

 protected:
    jblib::CudaArray<CudaFastFloat, 1>  dev_h_;
    jblib::CudaArray<CudaFastFloat, 1>  dev_mat_;
    jblib::CudaArray<CudaFastFloat, 1>  dev_s_float_;
    jblib::CudaArray<double, 1> dev_s_;
    jblib::CudaArray<double, 1> dev_s_new_;
    devDIA                      dev_J1ij_t_;

};

#endif  // JAMS_CORE_CUDASOLVER_H
