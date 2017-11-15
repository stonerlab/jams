// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDASOLVER_H
#define JAMS_CORE_CUDASOLVER_H

#include <cufft.h>

#include "cuda_defs.h"
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/physics.h"
#include "jams/core/monitor.h"
#include "jblib/containers/cuda_array.h"
#include "jblib/containers/vec.h"
#include "jams/cuda/wrappers/stream.h"


class CudaSolver : public Solver {
 public:
  CudaSolver() :
   dev_stream_()
   {};
  ~CudaSolver();

  void initialize(const libconfig::Setting& settings);
  void run();

  inline void notify_monitors() {
    bool is_device_synchonised = false;
    for (std::vector<Monitor*>::iterator it = monitors_.begin() ; it != monitors_.end(); ++it) {
      if((*it)->is_updating(iteration_)){
        if (!is_device_synchonised) {
          sync_device_data();
          is_device_synchonised = true;
        }
        (*it)->update(this);
      }
    }
  }

  inline double * dev_ptr_spin() {
    return dev_s_.data();
  }

  void compute_fields();

 protected:
    void sync_device_data();

    jblib::CudaArray<double, 1>  dev_h_;
    jblib::CudaArray<double, 1>  dev_gyro_;
    jblib::CudaArray<double, 1>  dev_alpha_;
    jblib::CudaArray<double, 1>  dev_s_;
    jblib::CudaArray<double, 1>  dev_s_old_;
    jblib::CudaArray<double, 1>  dev_ds_dt_;

  private:
    CudaStream dev_stream_;
};

#endif  // JAMS_CORE_CUDASOLVER_H
