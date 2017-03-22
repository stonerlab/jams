// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_H
#define JAMS_SOLVER_CUDA_HEUNLLG_H

#ifdef CUDA

#include <curand.h>
#include <cusparse.h>

#include "jams/core/cuda_solver.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class CUDAHeunLLGSolver : public CudaSolver {
  public:
    CUDAHeunLLGSolver() {};
    ~CUDAHeunLLGSolver();
    void initialize(int argc, char **argv, double dt);
    void run();

  private:
    cudaStream_t dev_stream_ = nullptr;
    jblib::CudaArray<CudaFastFloat, 1>  e_dev;
    jblib::Array<CudaFastFloat, 2> eng;
    int nblocks;
    int spmvblocksize;
};

#endif

#endif // JAMS_SOLVER_CUDA_HEUNLLG_H

