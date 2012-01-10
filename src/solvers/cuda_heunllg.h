#ifndef __CUDAHEUNLLG_H__
#define __CUDAHEUNLLG_H__

#include "solver.h"
#include "array.h"
#include "array2d.h"

#include <curand.h>
#include <cusparse.h>

struct devDIA {
  int     *row;
  int     *col;
  float   *val;
  size_t  pitch;
  int     blocks;
};

class CUDAHeunLLGSolver : public Solver {
  public:
    CUDAHeunLLGSolver()
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
    ~CUDAHeunLLGSolver();
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();
    void calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t);

  private:
    curandGenerator_t gen; // device random generator
    float * w_dev;
    cusparseHandle_t handle;
    cusparseMatDescr_t descra;
    devDIA  J1ij_s_dev;
    devDIA  J1ij_t_dev;
    devDIA  J2ij_s_dev;
    devDIA  J2ij_t_dev;
    double * s_dev;
    float  * sf_dev;
    double * s_new_dev;
    float * h_dev;
    float * e_dev;
    float * mat_dev;
    Array2D<float> eng;
    Array<double> sigma;
    int nblocks;
    int spmvblocksize;
};

#endif // __CUDAHEUNLLG_H__

