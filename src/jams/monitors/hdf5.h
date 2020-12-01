// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_HDF5_H
#define JAMS_MONITOR_HDF5_H

#include <iosfwd>
#include <string>

#include <libconfig.h++>

#include "jams/core/monitor.h"
#include "jams/helpers/slice.h"
#include "jams/interface/highfive.h"

class Solver;

class Hdf5Monitor : public Monitor {
 public:
  explicit Hdf5Monitor(const libconfig::Setting &settings);
  ~Hdf5Monitor();

  void update(Solver * solver) override;
    void post_process() override {};

    bool is_converged() override { return false; }

 private:
    void write_vector_field(const jams::MultiArray<double, 2>& field, const std::string& data_path, HighFive::File &file) const;
    void write_scalar_field(const jams::MultiArray<double, 1>& field, const std::string& data_path, HighFive::File &file) const;
  void open_new_xdmf_file(const std::string &xdmf_file_name);
  void update_xdmf_file(const std::string &h5_file_name);
  void write_lattice_h5_file(const std::string &h5_file_name);
  void write_spin_h5_file(const std::string &h5_file_name);

  bool         write_ds_dt_ = false;
  bool         compression_enabled_ = true;
  Slice        slice_;
  FILE*        xdmf_file_;
};

#endif  // JAMS_MONITOR_HDF5_H

