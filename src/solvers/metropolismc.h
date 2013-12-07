#ifndef __METROPOLISMC_H__
#define __METROPOLISMC_H__

#include "solver.h"

#include <containers/Array.h>

class MetropolisMCSolver : public Solver {
  public:
    MetropolisMCSolver() : snew(0,0), sigma(0,0), eng(0,0) {}
    ~MetropolisMCSolver() {}
    void initialise(int argc, char **argv, double dt);
    void run();
    void syncOutput();
    void calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s);

  private:
    jbLib::Array<double,2> snew;
    jbLib::Array<double,2> sigma;
    jbLib::Array<double,2> eng;

    void oneSpinEnergy(const int &i, double total[3]);
};

#endif // __METROPOLISMC_H__
