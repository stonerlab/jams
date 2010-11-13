#ifndef __BOLTZMANN_H__
#define __BOLTZMANN_H__

#include "monitor.h"
#include "array.h"
#include <fstream>

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
    void write();
  private:
    Array<double> bins;
    double total;
    std::ofstream outfile;
};

#endif // __BOLTZMANN_H__
