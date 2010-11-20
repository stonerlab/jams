#ifndef __CUDASEMILLG_H__
#define __CUDASEMILLG_H__

#include "solver.h"
#include "array2d.h"

#include <curand.h>

class CUDASemiLLGSolver : public Solver {
  public:
    CUDASemiLLGSolver(){};
    ~CUDASemiLLGSolver();
    void initialise(int argc, char **argv, double dt);
    void run();

  private:
    curandGenerator_t gen; // device random generator
    float * w_dev;

};

#endif // __CUDASEMILLG_H__

