// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_STRUCTUREFACTOR_H
#define JAMS_MONITOR_STRUCTUREFACTOR_H

#include <fstream>
#include <complex>
#include <vector>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/core/monitor.h"

#include "jblib/containers/vec.h"
#include "jblib/containers/array.h"

class Solver;

class StructureFactorMonitor : public Monitor {
 public:
  StructureFactorMonitor(const libconfig::Setting &settings);
  ~StructureFactorMonitor();

  void update(Solver * solver);
  bool is_converged() { return false; }
  std::string name() const {return "structurefactor";}

 private:

  void   fft_time();

    bool output_sublattice_enabled_ = false;
  jblib::Array<double, 2> s_transform;
  jblib::Array<double, 1> k0;
  jblib::Array<double, 1> kneq0;
  jblib::Array<std::complex<double>, 3> sqw_x;
  jblib::Array<std::complex<double>, 3> sqw_y;
  jblib::Array<std::complex<double>, 3> sqw_z;
  std::vector<jblib::Vec3<double> > bz_cfg_points;
  std::vector<jblib::Vec3<int> > bz_nodes;
  std::vector<jblib::Vec3<int> > bz_points;
  std::vector<double> bz_lengths;
  double freq_delta;
  int time_point_counter_;
};

#endif  // JAMS_MONITOR_STRUCTUREFACTOR_H

