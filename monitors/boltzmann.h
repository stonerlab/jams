// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_BOLTZMANN_H
#define JAMS_MONITOR_BOLTZMANN_H

#include <fstream>

#include "core/monitor.h"
#include "jblib/containers/array.h"

class BoltzmannMonitor : public Monitor{
 public:
  BoltzmannMonitor(const libconfig::Setting &settings);
  ~BoltzmannMonitor();

void update(Solver * solver);
bool is_converged() { return false; }
std::string name() const {return "boltzmann";}


 private:
  jblib::Array<double, 1> bins;
  double total;
  std::ofstream outfile;
};

#endif  // JAMS_MONITOR_BOLTZMANN_H
