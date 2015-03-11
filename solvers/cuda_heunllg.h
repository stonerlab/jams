// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_H
#define JAMS_SOLVER_CUDA_HEUNLLG_H

#ifdef CUDA

#include <curand.h>
#include <cusparse.h>

#include "core/cuda_solver.h"
#include "core/thermostat.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class CUDAHeunLLGSolver : public CudaSolver {
  public:
    CUDAHeunLLGSolver() {};
    ~CUDAHeunLLGSolver();
    void initialize(int argc, char **argv, double dt);
    void run();
    void compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s);

  private:
    Thermostat* thermostat_;

    jblib::CudaArray<CudaFastFloat, 1>  e_dev;
    jblib::Array<CudaFastFloat, 2> eng;
    int nblocks;
    int spmvblocksize;
};

#endif

#endif // JAMS_SOLVER_CUDA_HEUNLLG_H

