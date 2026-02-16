// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_HDF5_H
#define JAMS_MONITOR_HDF5_H

#include <jams/core/monitor.h>
#include <jams/helpers/slice.h>
#include <jams/interface/highfive.h>

#include <iosfwd>
#include <string>

class Solver;

class Hdf5Monitor : public Monitor {
 public:
  explicit Hdf5Monitor(const libconfig::Setting &settings);
  ~Hdf5Monitor() override;

  void update(Solver& solver) override;
  void post_process() override;

 private:
  void write_vector_field(const jams::MultiArray<double, 2>& field, const std::string& data_path, HighFive::File &file) const;
  void write_scalar_field(const jams::MultiArray<double, 1>& field, const std::string& data_path, HighFive::File &file) const;
  void open_new_xdmf_file(const std::string &xdmf_file_name);
  void update_xdmf_file(const std::string &h5_file_name, const double time);
  void write_lattice_h5_file(const std::string &h5_file_name);
  void write_spin_h5_file(const std::string &h5_file_name);
  void write_final_output();

  bool         write_ds_dt_ = false;
  bool         write_fields_ = false;
  bool         write_energies_ = false;
  bool         compression_enabled_ = false;
  bool         final_output_written_ = false;
  Slice        slice_;
  FILE*        xdmf_file_ = nullptr;
};

#endif  // JAMS_MONITOR_HDF5_H
