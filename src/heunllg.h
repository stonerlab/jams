#ifndef __HEUNLLG_H__
#define __HEUNLLG_H__

#include "solver.h"
#include "array2d.h"

class HeunLLGSolver : public Solver {
  public:
    HeunLLGSolver() : snew() {};
    ~HeunLLGSolver() {}
    void initialise(int argc, char **argv, double dt, NoiseType ntype);
    void run();

  private:
    Array2D<double> snew;
};

#endif // __HEUNLLG_H__
