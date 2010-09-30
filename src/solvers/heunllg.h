#ifndef __HEUNLLG_H__
#define __HEUNLLG_H__

#include "solver.h"
#include "vecfield.h"

class HeunLLGSolver : public Solver {
  public:

    void initialise(int argc, char **argv, double dt);
    void run();

  private:
    vecField snew;
}

#endif // __HEUNLLG_H__
