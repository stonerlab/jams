// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CUDA_SOLVER_H
#define JAMS_CUDA_SOLVER_H

#ifdef CUDA

#include <curand.h>
#include <cusparse.h>

#include "core/cuda_sparse_types.h"
#include "core/solver.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class CudaSolver : public Solver {
  public:
    CudaSolver(){};
    ~CudaSolver();
    void initialize(int argc, char **argv, double dt);
    void run();

    void compute_fields();
    void compute_energy();

    void sync_device_data();

  private:
};

#endif

#endif // JAMS_SOLVER_CUDA_HEUNLLG_H

