#ifndef JAMS_CORE_MONITOR_H
#define JAMS_CORE_MONITOR_H

#include "solver.h"


enum ConvergenceType {convNone, convMag, convPhi, convSinPhi};

class Monitor {
  public:
    Monitor() : initialised(false) {}

    virtual ~Monitor(){}

    virtual void initialise();
    virtual void run();
    virtual void write(Solver *solver);
	
	virtual void initConvergence(ConvergenceType type, const double meanTol, const double devTol);
	virtual bool checkConvergence();
	

    static Monitor* Create();
  protected:
    bool initialised;

};

#endif // JAMS_CORE_MONITOR_H
