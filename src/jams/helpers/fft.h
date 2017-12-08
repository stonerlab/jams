// Copyright 2016 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_FFT_H
#define JAMS_MONITOR_FFT_H

#include <complex>
#include <jblib/containers/array.h>

// Windowing functions
double fft_window_default(const int n, const int n_total);
double fft_window_hanning(const int n, const int n_total);
double fft_window_blackman_4(const int n, const int n_total);

fftw_plan fft_plan_rspace_to_kspace(double * rspace, std::complex<double> * kspace, const Vec3i& kspace_size, const int & motif_size);

void apply_kspace_phase_factors(jblib::Array<std::complex<double>, 5> &kspace);

#endif  // JAMS_MONITOR_FFT_H

