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

  void update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field);

 private:

  void   fft_time();
  double fft_windowing(const int n, const int n_total);

  std::ofstream outfile;
  fftw_plan fft_plan_sq_xy;
  jblib::Array<fftw_complex, 3> sq_xy;
  std::vector<std::complex<double> > sqw;
};

#endif  // JAMS_MONITOR_STRUCTUREFACTOR_H

