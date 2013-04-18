#ifndef __ENERGY_H__
#define __ENERGY_H__

#include "monitor.h"
#include "array2d.h"
#include <fstream>

class EnergyMonitor : public Monitor {
  public:
    EnergyMonitor(){};

    ~EnergyMonitor();

    void initialise();
    void run();
    void write(Solver *solver);
	void initConvergence(ConvergenceType type, const double meanTol, const double devTol);
	bool checkConvergence();
  private:
    std::ofstream outfile;
};

#endif // __ENERGY_H__

