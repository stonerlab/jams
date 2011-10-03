#ifndef __CUDAHEUNLLG_H__
#define __CUDAHEUNLLG_H__

#include "solver.h"
#include "array.h"
#include "array2d.h"

#include <curand.h>
#include <cusparse.h>

class CUDAHeunLLGSolver : public Solver {
  public:
    CUDAHeunLLGSolver()
      : gen(0),
        w_dev(0),
        handle(0),
        descra(0),
        Jij_dev_row(0),
        Jij_dev_col(0),
        Jij_dev_val(0),
        J2ij_dev_row(0),
        J2ij_dev_col(0),
        J2ij_dev_val(0),
        s_dev(0),
        sf_dev(0),
        s_new_dev(0),
        h_dev(0),
        mat_dev(0),
        sigma(0),
        nblocks(0),
        spmvblocksize(0),
        Jspmvblocks(0),
        J2spmvblocks(0),
        JdiaPitch(0),
        J2diaPitch(0)
    {};
    ~CUDAHeunLLGSolver();
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();

  private:
    curandGenerator_t gen; // device random generator
    float * w_dev;
    cusparseHandle_t handle;
    cusparseMatDescr_t descra;
    int * Jij_dev_row;
    int * Jij_dev_col;
    float * Jij_dev_val;
    int * J2ij_dev_row;
    int * J2ij_dev_col;
    float * J2ij_dev_val;
    double * s_dev;
    float  * sf_dev;
    double * s_new_dev;
    float * h_dev;
    float * mat_dev;
    Array<double> sigma;
    int nblocks;
    int spmvblocksize;
    int Jspmvblocks;
    int J2spmvblocks;
    size_t JdiaPitch;
    size_t J2diaPitch;
};

#endif // __CUDAHEUNLLG_H__

