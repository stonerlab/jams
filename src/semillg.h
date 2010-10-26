#ifndef __SEMILLG_H__
#define __SEMILLG_H__

#include "solver.h"
#include "array2d.h"

class SemiLLGSolver : public Solver {
  public:
    SemiLLGSolver() : sold(), sigma() {};
    ~SemiLLGSolver() {}
    void initialise(int argc, char **argv, double dt);
    void fields();
    void run();

  private:
    Array2D<double> sold;
    Array2D<double> sigma;
};

#endif // __SEMILLG_H__
