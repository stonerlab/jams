// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_HDF5_H
#define JAMS_MONITOR_HDF5_H

#include <vector>

#include "H5Cpp.h"  // NOLINT

#include "core/monitor.h"
#include "core/runningstat.h"
#include "core/slice.h"

#include "jblib/containers/array.h"

class Hdf5Monitor : public Monitor {
 public:
  explicit Hdf5Monitor(const libconfig::Setting &settings);
  ~Hdf5Monitor();

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:
  void output_lattice();

  H5::PredType float_pred_type;
  bool         is_compression_enabled;
  Slice        slice;
};

#endif  // JAMS_MONITOR_HDF5_H

