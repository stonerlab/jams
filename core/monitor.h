// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_MONITOR_H
#define JAMS_CORE_MONITOR_H

#include "core/solver.h"

enum ConvergenceType {convNone, convMag, convPhi, convSinPhi};

class Monitor {
 public:
  Monitor() : initialized(false) {}

  virtual ~Monitor() {}

  virtual void initialize();
  virtual void run();
  virtual void write(Solver *solver);

  virtual void initialize_convergence(ConvergenceType type, const double meanTol,
    const double devTol);
  virtual bool has_converged();


  static Monitor* Create();
 protected:
    bool initialized;
};

#endif  // JAMS_CORE_MONITOR_H
