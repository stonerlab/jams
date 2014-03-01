// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_ANISOTROPY_ENERGY_H
#define JAMS_MONITOR_ANISOTROPY_ENERGY_H

#include <fstream>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class AnisotropyEnergyMonitor : public Monitor {
 public:
  AnisotropyEnergyMonitor(const libconfig::Setting &settings);
  ~AnisotropyEnergyMonitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:
  jblib::Array<double, 2> dz_energy_;
  std::ofstream outfile;
};

#endif  // JAMS_MONITOR_ANISOTROPY_ENERGY_H

