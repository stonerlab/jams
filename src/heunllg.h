#ifndef __HEUNLLG_H__
#define __HEUNLLG_H__

#include "solver.h"
#include "array2d.h"

class HeunLLGSolver : public Solver {
  public:
    HeunLLGSolver() : snew(0,0), sigma(0,0) {};
    ~HeunLLGSolver() {}
    void initialise(int argc, char **argv, double dt);
    void run();

  private:
    Array2D<double> snew;
    Array2D<double> sigma;
};

#endif // __HEUNLLG_H__
