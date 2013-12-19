#ifndef __BOLTZMANN_MAG_H__
#define __BOLTZMANN_MAG_H__

#include "monitor.h"
#include <fstream>
#include <containers/array.h>

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
    jblib::Array<double,1> bins;
    double total;
    std::ofstream outfile;
};

#endif // __BOLTZMANN_MAG_H__
