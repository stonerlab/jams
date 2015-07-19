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

  void update(const Solver * const solver);

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
  std::vector<std::complex<double> > sqw_x;
  std::vector<std::complex<double> > sqw_y;
  std::vector<std::complex<double> > sqw_z;
  std::vector<jblib::Vec3<int> > bz_points;
  std::vector<double> bz_lengths;
  double delta_freq;
};

#endif  // JAMS_MONITOR_STRUCTUREFACTOR_H

