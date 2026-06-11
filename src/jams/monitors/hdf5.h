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
  ~Hdf5Monitor();

  void update(Solver& solver) override;
    void post_process() override {};

 private:
    void open_new_xdmf_file(const std::string &xdmf_file_name);
    void update_xdmf_file(const std::string &h5_file_name, const double time);
    void write_lattice_h5_file(const std::string &h5_file_name);
    void write_spin_h5_file(const std::string &h5_file_name);
    [[nodiscard]] int output_point_count();
    [[nodiscard]] int source_spin_index(int output_index);
    void write_xdmf_scalar_attribute(
        const std::string& name,
        const std::string& h5_file_name,
        const std::string& data_path,
        unsigned data_dimension,
        unsigned precision,
        const std::string& number_type);
    void write_xdmf_vector_attribute(
        const std::string& name,
        const std::string& h5_file_name,
        const std::string& data_path,
        unsigned data_dimension,
        unsigned precision);

  bool         write_ds_dt_ = false;
  bool         compression_enabled_ = true;
  Slice        slice_;
  FILE*        xdmf_file_;
};

#endif  // JAMS_MONITOR_HDF5_H
