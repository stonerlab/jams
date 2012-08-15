#ifndef __MAGNETISATION_H__
#define __MAGNETISATION_H__

#include "monitor.h"
#include "array2d.h"
#include "runningstat.h"
#include <fstream>

class MagnetisationMonitor : public Monitor {
  public:
    MagnetisationMonitor()
      : mag(0,0),
        outfile(),
		convType(convNone),
		tolerance(1E10),
		rs()
    {}

    ~MagnetisationMonitor();

    void initialise();
    void run();
    void write(const double &time);
	void initConvergence(ConvergenceType type, const double tol);
	bool checkConvergence();
  private:
    Array2D<double> mag;
    std::ofstream outfile;
	ConvergenceType convType;
	double			tolerance;
	RunningStat		rs;
};

#endif // __MAGNETISATION_H__

