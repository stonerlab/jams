#ifndef __BOLTZMANN_H__
#define __BOLTZMANN_H__

#include "monitor.h"
#include <fstream>
#include <containers/Array.h>

class BoltzmannMonitor : public Monitor{
  public:
    BoltzmannMonitor()
      : bins(0),
        total(0),
        outfile()
    {}
    ~BoltzmannMonitor();

    void initialise();
    void run();
    void write(Solver *solver);
  private:
    jbLib::Array<double,1> bins;
    double total;
    std::ofstream outfile;
};

#endif // __BOLTZMANN_H__
