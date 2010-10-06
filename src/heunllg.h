#ifndef __HEUNLLG_H__
#define __HEUNLLG_H__

#include "solver.h"
#include "vecfield.h"

class HeunLLGSolver : public Solver {
  public:
    HeunLLGSolver();
    ~HeunLLGSolver();
    void initialise(int argc, char **argv, double dt);
    void run();

  private:
    VecField<double> snew;
};

#endif // __HEUNLLG_H__
