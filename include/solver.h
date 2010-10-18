#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "globals.h"

enum SolverType{ HEUNLLG };

class Solver 
{
  public:
    Solver() 
      : initialised(false),
        time(0.0),
        iteration(0),
        dt(0.0)
      {}

    virtual ~Solver(){}

    virtual void initialise(int argc, char **argv, double dt);
    virtual void run();

    static Solver* Create();
    static Solver* Create(SolverType type);
  protected:
    bool initialised;

    double time;  // current time

    int iteration; // number of iterations
    double dt;

};

#endif // __SOLVER_H__
