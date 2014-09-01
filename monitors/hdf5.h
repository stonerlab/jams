// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_HDF5_H
#define JAMS_MONITOR_HDF5_H

#include <fstream>
#include <vector>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class Hdf5Monitor : public Monitor {
 public:
  Hdf5Monitor(const libconfig::Setting &settings);
  ~Hdf5Monitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:

};

#endif  // JAMS_MONITOR_HDF5_H

