#ifndef __SOLVER_H__
#define __SOLVER_H__


#include "globals.h"

enum SolverType{ HEUNLLG, HEUNLLMS, SEMILLG, FFTNOISE };


class Solver 
{
  public:
    Solver() 
      : initialised(false),
        time(0.0),
        iteration(0),
        temperature(0),
        dt(0.0),
        t_step(0.0)
      {}

    virtual ~Solver(){}

    virtual void initialise(int argc, char **argv, double dt);
    virtual void run();

    inline int getIteration() { return iteration; }
    inline double getTime() { return iteration*t_step; }
    inline double getTemperature() { return temperature; }
    inline void setTemperature(double &t) { temperature = t; }

    static Solver* Create();
    static Solver* Create(SolverType type);
  protected:
    bool initialised;

    double time;  // current time

    int iteration; // number of iterations

    double temperature;
    double dt;
    double t_step;

};

#endif // __SOLVER_H__
