#ifndef JAMS_MONITOR_ENERGY_H
#define JAMS_MONITOR_ENERGY_H

#include "monitor.h"
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

#endif // JAMS_MONITOR_ENERGY_H

