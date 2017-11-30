// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDASOLVER_H
#define JAMS_CORE_CUDASOLVER_H

#include "jams/core/solver.h"
#include "jblib/containers/cuda_array.h"

class CudaSolver : public Solver {
 public:
  CudaSolver() = default;
  ~CudaSolver() = default;

  void initialize(const libconfig::Setting& settings);
  virtual void run() = 0;

  void notify_monitors();

  double * dev_ptr_spin();

  void compute_fields();

 protected:
    void sync_device_data();

    jblib::CudaArray<double, 1>  dev_h_;
    jblib::CudaArray<double, 1>  dev_gyro_;
    jblib::CudaArray<double, 1>  dev_alpha_;
    jblib::CudaArray<double, 1>  dev_s_;
    jblib::CudaArray<double, 1>  dev_s_old_;
    jblib::CudaArray<double, 1>  dev_ds_dt_;
};

#endif  // JAMS_CORE_CUDASOLVER_H
