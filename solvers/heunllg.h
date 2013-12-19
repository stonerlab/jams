#ifndef __HEUNLLG_H__
#define __HEUNLLG_H__

#include "solver.h"
#include <containers/array.h>

class HeunLLGSolver : public Solver {
  public:
    HeunLLGSolver() : snew(0,0), sigma(0,0), eng(0,0) {}
    ~HeunLLGSolver() {}
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();
    void calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s);

  private:
    jblib::Array<double,2> snew;
    jblib::Array<double,2> sigma;
    jblib::Array<double,2> eng;
};

#endif // __HEUNLLG_H__
