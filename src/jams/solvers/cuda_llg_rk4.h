// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_LLG_RK4_H
#define JAMS_SOLVER_CUDA_LLG_RK4_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"

class CUDALLGRK4Solver : public CudaSolver {
  public:
    CUDALLGRK4Solver() = default;
    ~CUDALLGRK4Solver() = default;
    void initialize(const libconfig::Setting& settings);
    void run();

  private:
    CudaStream dev_stream_;
    bool zero_safe_kernels_required_;
    double dt_;
    jams::MultiArray<double, 2> s_old_;
    jams::MultiArray<double, 2> k1_;
    jams::MultiArray<double, 2> k2_;
    jams::MultiArray<double, 2> k3_;
    jams::MultiArray<double, 2> k4_;
};

#endif

#endif // JAMS_SOLVER_CUDA_LLG_RK4_H

