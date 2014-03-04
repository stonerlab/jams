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
    bool is_device_synchonised = false;
    for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
      if((*it)->is_updating(iteration_)){
        if (!is_device_synchonised) {
          sync_device_data();
          is_device_synchonised = true;
        }
        (*it)->update(iteration_, time(), physics_module_->temperature(), physics_module_->applied_field());
      }
    }
  }



  void compute_fields();
  void compute_energy();

 protected:
    inline void sync_device_data() {
      dev_s_.copy_to_host_array(globals::s);
    }

    jblib::CudaArray<CudaFastFloat, 1>  dev_h_;
    jblib::CudaArray<CudaFastFloat, 1>  dev_mat_;
    jblib::CudaArray<CudaFastFloat, 1>  dev_s_float_;
    jblib::CudaArray<double, 1> dev_s_;
    jblib::CudaArray<double, 1> dev_s_new_;
    devDIA                      dev_J1ij_t_;
    jblib::CudaArray<CudaFastFloat, 1> dev_d2z_;
    jblib::CudaArray<CudaFastFloat, 1> dev_d4z_;
    jblib::CudaArray<CudaFastFloat, 1> dev_d6z_;

};

#endif  // JAMS_CORE_CUDASOLVER_H
