#ifndef __CUDASEMILLG_H__
#define __CUDASEMILLG_H__

#include "solver.h"
#include "array2d.h"

#include <curand.h>
#include <cusparse.h>

class CUDASemiLLGSolver : public Solver {
  public:
    CUDASemiLLGSolver()
      : gen(0),
        w_dev(0),
        handle(0),
        descra(0),
        Jij_dev_row(0),
        Jij_dev_col(0),
        Jij_dev_val(0),
        s_dev(0),
        s_new_dev(0),
        h_dev(0),
        h_new_dev(0),
        mus_dev(0),
        gyro_dev(0),
        alpha_dev(0)
    {};
    ~CUDASemiLLGSolver();
    void initialise(int argc, char **argv, double dt);
    void run();

  private:
    curandGenerator_t gen; // device random generator
    float * w_dev;
    cusparseHandle_t handle;
    cusparseMatDescr_t descra;
    int * Jij_dev_row;
    int * Jij_dev_col;
    double * Jij_dev_val;
    double * s_dev;
    double * s_new_dev;
    double * h_dev;
    double * h_new_dev;
    double * mus_dev;
    double * gyro_dev;
    double * alpha_dev;
};

#endif // __CUDASEMILLG_H__
