#ifndef JAMS_MONITOR_MAGNETISATION_H
#define JAMS_MONITOR_MAGNETISATION_H

#include <fstream>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class MagnetisationMonitor : public Monitor {
  public:
    MagnetisationMonitor()
      : mag(0,0),
        outfile(),
		convType(convNone),
		meanTol(1E10),
		devTol(1E10),
		blockStats(),
		runningMean(),
		old_avg(0.0)
    {}

    ~MagnetisationMonitor();

    void initialise();
    void run();
    void write(Solver *solver);
	void initConvergence(ConvergenceType type, const double meanTol, const double devTol);
	bool checkConvergence();
  private:
    jblib::Array<double,2> mag;
    std::ofstream outfile;
	ConvergenceType convType;
	double			meanTol;
	double			devTol;
	RunningStat		blockStats;
	RunningStat		runningMean;
    double      old_avg;
};

#endif // JAMS_MONITOR_MAGNETISATION_H

