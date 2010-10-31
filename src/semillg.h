#ifndef __SEMILLG_H__
#define __SEMILLG_H__

#include "solver.h"
#include "array2d.h"

class SemiLLGSolver : public Solver {
  public:
    SemiLLGSolver() : sold() {};
    ~SemiLLGSolver() {}
    void initialise(int argc, char **argv, double dt, NoiseType ntype);
    void run();

  private:
    Array2D<double> sold;
};

#endif // __SEMILLG_H__
