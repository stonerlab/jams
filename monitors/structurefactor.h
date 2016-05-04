// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_STRUCTUREFACTOR_H
#define JAMS_MONITOR_STRUCTUREFACTOR_H

#include <fstream>
#include <complex>

#include "core/monitor.h"
#include "core/runningstat.h"

#include "jblib/containers/array.h"

class StructureFactorMonitor : public Monitor {
 public:
  StructureFactorMonitor(const libconfig::Setting &settings);
  ~StructureFactorMonitor();

  void update(Solver * solver);
  bool is_converged() { return false; }
  std::string name() const {return "structurefactor";}

 private:

  void   fft_time();
  double fft_windowing(const int n, const int n_total);

  fftw_plan fft_plan_sq_x;
  fftw_plan fft_plan_sq_y;
  fftw_plan fft_plan_sq_z;
  jblib::Array<fftw_complex, 3> sq_x;
  jblib::Array<fftw_complex, 3> sq_y;
  jblib::Array<fftw_complex, 3> sq_z;
  jblib::Array<double, 2> s_transform;
  jblib::Array<std::complex<double>, 3> sqw_x;
  jblib::Array<std::complex<double>, 3> sqw_y;
  jblib::Array<std::complex<double>, 3> sqw_z;
  jblib::Array<fftw_complex, 1> chi_xy;
  jblib::Array<fftw_complex, 1> chi_yx;
  std::vector<jblib::Vec3<double> > bz_cfg_points;
  std::vector<jblib::Vec3<int> > bz_nodes;
  std::vector<jblib::Vec3<int> > bz_points;
  std::vector<double> bz_lengths;
  double freq_delta;
  int time_point_counter_;
};

#endif  // JAMS_MONITOR_STRUCTUREFACTOR_H

