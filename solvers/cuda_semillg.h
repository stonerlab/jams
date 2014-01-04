#ifndef JAMS_SOLVER_CUDA_SEMILLG_H
#define JAMS_SOLVER_CUDA_SEMILLG_H

#ifdef CUDA

#include "solver.h"
#include "cuda_sparse_types.h"

#include <curand.h>
#include <cusparse.h>

#include <containers/array.h>

class CUDASemiLLGSolver : public Solver {
  public:
    CUDASemiLLGSolver()
      : gen(0),
        w_dev(0),
        handle(0),
        descra(0),
        J1ij_s_dev(),
        J1ij_t_dev(),
        J2ij_s_dev(),
        J2ij_t_dev(),
        s_dev(0),
        sf_dev(0),
        s_new_dev(0),
        h_dev(0),
        e_dev(0),
        mat_dev(0),
        eng(0,0),
        sigma(0),
        nblocks(0),
        spmvblocksize(0)
    {};
    ~CUDASemiLLGSolver();
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();

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
    float  * sf_dev;
    double * s_new_dev;
    float * h_dev;
    float * e_dev;
    float * mat_dev;
    jblib::Array<float,2> eng;
    jblib::Array<double,1> sigma;
    int nblocks;
    int spmvblocksize;
};

#endif

#endif // JAMS_SOLVER_CUDA_SEMILLG_H

