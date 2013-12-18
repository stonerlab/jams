#ifndef __CUDAHEUNLLBP_H__
#define __CUDAHEUNLLBP_H__

#ifdef CUDA

#include "solver.h"
#include "cuda_sparse_types.h"

#include <curand.h>
#include <cusparse.h>

#include <containers/Array.h>

class CUDAHeunLLBPSolver : public Solver {
  public:
    CUDAHeunLLBPSolver()
      : gen(0),
        w_dev(0),
		tc_dev(0),	
        handle(0),
        descra(0),
        J1ij_s_dev(),
        J1ij_t_dev(),
        J2ij_s_dev(),
        J2ij_t_dev(),
        s_dev(0),
        sf_dev(0),
        s_new_dev(0),
		u1_dev(0),
		u2_dev(0),
		u1_new_dev(0),
		u2_new_dev(0),
        h_dev(0),
        mat_dev(0),
        sigma(0),
		t_corr(0,0),
        nblocks(0),
        spmvblocksize(0)
    {};
    ~CUDAHeunLLBPSolver();
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();
    void calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s);

  private:
    curandGenerator_t gen; // device random generator
    float * w_dev;    				
	float * tc_dev; 				// store tc_1 and tc_2 in a 2D array
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
	double * u1_dev;
	double * u2_dev;
	double * u1_new_dev;
	double * u2_new_dev;
    float * h_dev;
    float * mat_dev;
    jbLib::Array<double,1> sigma;
    jbLib::Array<float,2> t_corr;
    int nblocks;
    int spmvblocksize;
};

#endif

#endif // __CUDAHEUNLLBP_H__

