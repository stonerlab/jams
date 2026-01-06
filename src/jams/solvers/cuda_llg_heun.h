// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_H
#define JAMS_SOLVER_CUDA_HEUNLLG_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDAHeunLLGSolver : public CudaSolver {
  public:
    CUDAHeunLLGSolver() = default;
    ~CUDAHeunLLGSolver() override = default;

    inline explicit CUDAHeunLLGSolver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-heun-gpu"; }

  private:
    CudaStream dev_stream_;
    jams::MultiArray<double, 2> s_old_;
};

#endif

#endif // JAMS_SOLVER_CUDA_HEUNLLG_H

