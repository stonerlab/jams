#ifndef __BOLTZMANN_MAG_H__
#define __BOLTZMANN_MAG_H__

#include "monitor.h"
#include <fstream>
#include <containers/Array.h>

class BoltzmannMagMonitor : public Monitor{
  public:
    BoltzmannMagMonitor()
      : bins(0),
        total(0),
        outfile()
    {}
    ~BoltzmannMagMonitor();

    void initialise();
    void run();
    void write(Solver *solver);
  private:
    jbLib::Array<double,1> bins;
    double total;
    std::ofstream outfile;
};

#endif // __BOLTZMANN_MAG_H__
