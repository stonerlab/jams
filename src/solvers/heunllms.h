#ifndef __HEUNLLMS_H__
#define __HEUNLLMS_H__

#include "solver.h"
#include "array.h"
#include "array2d.h"

class HeunLLMSSolver : public Solver {
  public:
    HeunLLMSSolver() : snew(0,0), wnew(0,0), u(0,0), sigma(0,0) {}
    ~HeunLLMSSolver() {}
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();

  private:
    Array2D<double> snew;
    Array2D<double> wnew;
    Array2D<double> u;
    Array2D<double> sigma;
};

#endif // __HEUNLLG_H__
