#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "globals.h"

class Solver
{
  public:
    Solver();
    virtual ~Solver(){}

    virtual void initialise(int argc, char **argv, double dt);
    virtual void run();
  private:
    bool initialised;

    double time;  // current time

    int iteration; // number of iterations

}

#endif // __SOLVER_H__
