#ifndef __CUDAHEUNLLG_H__
#define __CUDAHEUNLLG_H__

#ifdef CUDA

#include "solver.h"
#include "cuda_sparse_types.h"

#include <curand.h>
#include <cusparse.h>

#include <containers/array.h>
#include <containers/cuda_array.h>

class CUDAHeunLLGSolver : public Solver {
  public:
    CUDAHeunLLGSolver()
      : gen(0),
        handle(0),
        descra(0),
        J1ij_s_dev(),
        J1ij_t_dev(),
        J2ij_s_dev(),
        J2ij_t_dev(),
        w_dev(),
        r_dev(),
        e_dev(),
        h_dev(),
        mat_dev(),
        h_dipole_dev(),
        r_max_dev(),
        pbc_dev(),
        sf_dev(),
        s_dev(),
        s_new_dev(),
        eng(0,0),
        sigma(0),
        nblocks(0),
        spmvblocksize(0)
    {};
    ~CUDAHeunLLGSolver();
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();
    void calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s);

  private:
    curandGenerator_t gen; // device random generator
    cusparseHandle_t handle;
    cusparseMatDescr_t descra;
    devDIA  J1ij_s_dev;
    devDIA  J1ij_t_dev;
    devDIA  J2ij_s_dev;
    devDIA  J2ij_t_dev;
    devCSR  J4ijkl_s_dev;
    jblib::CudaArray<float,1>  w_dev;
    jblib::CudaArray<float,1>  r_dev;
    jblib::CudaArray<float,1>  e_dev;
    jblib::CudaArray<float,1>  h_dev;
    jblib::CudaArray<float,1>  mat_dev;
    jblib::CudaArray<float,1>  h_dipole_dev;
    jblib::CudaArray<float,1>  r_max_dev;
    jblib::CudaArray<bool,1>   pbc_dev;
    jblib::CudaArray<float,1>  sf_dev;
    jblib::CudaArray<double,1> s_dev;
    jblib::CudaArray<double,1> s_new_dev;
    jblib::Array<float,2> eng;
    jblib::Array<double,1> sigma;
    int nblocks;
    int spmvblocksize;
};

#endif

#endif // __CUDAHEUNLLG_H__

