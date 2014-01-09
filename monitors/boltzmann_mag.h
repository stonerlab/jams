// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_BOLTZMANN_MAG_H
#define JAMS_MONITOR_BOLTZMANN_MAG_H

#include <fstream>

#include "core/monitor.h"
#include "jblib/containers/array.h"

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
  jblib::Array<double, 1> bins;
  double total;
  std::ofstream outfile;
};

#endif  // JAMS_MONITOR_BOLTZMANN_MAG_H
