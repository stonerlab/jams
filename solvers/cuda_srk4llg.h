// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_SOLVER_CUDA_SRK4LLG_H
#define JAMS_SOLVER_CUDA_SRK4LLG_H

#ifdef CUDA

#include <curand.h>
#include <cusparse.h>

#include "core/cuda_sparse_types.h"
#include "core/solver.h"

#include "jblib/containers/array.h"

class CUDALLGSolverSRK4 : public Solver {
  public:
    CUDALLGSolverSRK4()
      : gen(0),
        w_dev(0),
        handle(0),
        descra(0),
        J1ij_s_dev(),
        J1ij_t_dev(),
        J2ij_s_dev(),
        J2ij_t_dev(),
        s_dev(0),
        s_old_dev(0),
        k0_dev(0),
        k1_dev(0),
        k2_dev(0),
        sf_dev(0),
        r_dev(0),
        r_max_dev(0),
        pbc_dev(0),
        h_dev(0),
        h_dipole_dev(0),
        e_dev(0),
        mat_dev(0),
        eng(0, 0),
        sigma(0),
        nblocks(0),
        spmvblocksize(0)
    {};
    ~CUDALLGSolverSRK4();
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();
    void calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s);

  private:
    curandGenerator_t gen; // device random generator
    float * w_dev;
    cusparseHandle_t handle;
    cusparseMatDescr_t descra;
    devDIA  J1ij_s_dev;
    devDIA  J1ij_t_dev;
    devDIA  J2ij_s_dev;
    devDIA  J2ij_t_dev;
    devCSR  J4ijkl_s_dev;
    double * s_dev;
    double * s_old_dev;
    double * k0_dev;
    double * k1_dev;
    double * k2_dev;
    float  * sf_dev;
    float * r_dev;
    float * r_max_dev;
    bool * pbc_dev;
    float * h_dev;
    float * h_dipole_dev;
    float * e_dev;
    float * mat_dev;
    jblib::Array<float, 2> eng;
    jblib::Array<double, 1> sigma;
    int nblocks;
    int spmvblocksize;
};

#endif

#endif // JAMS_SOLVER_CUDA_SRK4LLG_H

