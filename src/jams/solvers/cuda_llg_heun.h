// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_H
#define JAMS_SOLVER_CUDA_HEUNLLG_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"

class CUDAHeunLLGSolver : public CudaSolver {
  public:
    CUDAHeunLLGSolver() = default;
    ~CUDAHeunLLGSolver() = default;
    void initialize(const libconfig::Setting& settings);
    void run();

  private:
    CudaStream dev_stream_;
    bool zero_safe_kernels_required_;
    double dt_;
    jams::MultiArray<double, 2> s_old_;
};

#endif

#endif // JAMS_SOLVER_CUDA_HEUNLLG_H

