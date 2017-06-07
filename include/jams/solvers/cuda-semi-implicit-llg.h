// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_SEMI_IMPLICIT_LLG_H
#define JAMS_SOLVER_CUDA_SEMI_IMPLICIT_LLG_H

#ifdef CUDA

#include <curand.h>
#include <cusparse.h>

#include "jams/core/cuda_solver.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class CUDASemiImplicitLLGSolver : public CudaSolver {
  public:
    CUDASemiImplicitLLGSolver() {};
    ~CUDASemiImplicitLLGSolver();
    void initialize(int argc, char **argv, double dt);
    void run();

  private:
    cudaStream_t dev_stream_ = nullptr;
};

#endif

#endif // JAMS_SOLVER_CUDA_SEMI_IMPLICIT_LLG_H

