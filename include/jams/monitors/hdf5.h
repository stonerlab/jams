// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_HDF5_H
#define JAMS_MONITOR_HDF5_H

#include <vector>

#include "H5Cpp.h"  // NOLINT

#include "jams/core/monitor.h"
#include "jams/core/runningstat.h"
#include "jams/core/slice.h"

#include "jblib/containers/array.h"

class Hdf5Monitor : public Monitor {
 public:
  explicit Hdf5Monitor(const libconfig::Setting &settings);
  ~Hdf5Monitor();

  void update(Solver * solver);
  bool is_converged() { return false; }
  std::string name() const {return "hdf5";}

 private:
  void open_new_xdmf_file(const std::string &xdmf_file_name);
  void update_xdmf_file(const std::string &h5_file_name, const H5::PredType float_type);
  void write_lattice_h5_file(const std::string &h5_file_name, const H5::PredType float_type);
  void write_spin_h5_file(const std::string &h5_file_name, const H5::PredType float_type);

  H5::PredType float_pred_type_;
  bool         compression_enabled_;
  Slice        slice_;
  FILE*        xdmf_file_;
};

#endif  // JAMS_MONITOR_HDF5_H

