// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_HEUNLLG_H
#define JAMS_SOLVER_CUDA_HEUNLLG_H

#ifdef CUDA

#include <curand.h>
#include <cusparse.h>

#include "core/cuda_sparse_types.h"
#include "core/solver.h"

#include "jblib/containers/array.h"
#include "jblib/containers/cuda_array.h"

class CUDAHeunLLGSolver : public Solver {
  public:
    CUDAHeunLLGSolver()
      : gen(0),
        handle(0),
        descra(0),
        J1ij_t_dev(),
        w_dev(),
        e_dev(),
        h_dev(),
        mat_dev(),
        sf_dev(),
        s_dev(),
        s_new_dev(),
        eng(0, 0),
        sigma(0),
        nblocks(0),
        spmvblocksize(0)
    {};
    ~CUDAHeunLLGSolver();
    void initialize(int argc, char **argv, double dt);
    void run();
    void sync_device_data();
    void compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s);

  private:
    curandGenerator_t gen; // device random generator
    cusparseHandle_t handle;
    cusparseMatDescr_t descra;
    devDIA  J1ij_t_dev;
    jblib::CudaArray<float, 1>  w_dev;
    jblib::CudaArray<float, 1>  e_dev;
    jblib::CudaArray<float, 1>  h_dev;
    jblib::CudaArray<float, 1>  mat_dev;
    jblib::CudaArray<float, 1>  sf_dev;
    jblib::CudaArray<double, 1> s_dev;
    jblib::CudaArray<double, 1> s_new_dev;
    jblib::Array<float, 2> eng;
    jblib::Array<double, 1> sigma;
    int nblocks;
    int spmvblocksize;
};

#endif

#endif // JAMS_SOLVER_CUDA_HEUNLLG_H

