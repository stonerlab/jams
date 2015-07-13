// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDASOLVER_H
#define JAMS_CORE_CUDASOLVER_H

#include <cufft.h>

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
        (*it)->update(this);
      }
    }
  }

  inline double * dev_ptr_spin() {
    return dev_s_.data();
  }

  void compute_fields();
  void compute_energy();

 protected:
    inline void sync_device_data() {
      dev_s_.copy_to_host_array(globals::s);
    }

    jblib::Vec3<int> num_kpoints_;
    jblib::CudaArray<int, 1> r_to_k_mapping_;

    jblib::CudaArray<double, 1>  dev_h_;
    jblib::CudaArray<CudaFastFloat, 1>  dev_mat_;
    jblib::CudaArray<double, 1> dev_s_;
    jblib::CudaArray<double, 1> dev_s_new_;

    jblib::CudaArray<double, 1> dev_s3d_;
    jblib::CudaArray<double, 1> dev_h3d_;

    jblib::CudaArray<cufftDoubleComplex, 1> dev_sq_;
    jblib::CudaArray<cufftDoubleComplex, 1> dev_hq_;
    jblib::CudaArray<cufftDoubleComplex, 1> dev_wq_;

    cufftHandle spin_fft_forward_transform;
    cufftHandle field_fft_backward_transform;

    cudaStream_t* dev_streams_;
};

#endif  // JAMS_CORE_CUDASOLVER_H
