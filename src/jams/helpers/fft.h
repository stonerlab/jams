// Copyright 2016 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_FFT_H
#define JAMS_MONITOR_FFT_H

#include <complex>
#include <array>
#include <fftw3.h>

#include "jams/containers/multiarray.h"

#define FFTW_COMPLEX_CAST(x) reinterpret_cast<fftw_complex*>(x)

// Windowing functions
double fft_window_default(const int n, const int n_total);
double fft_window_hanning(const int n, const int n_total);
double fft_window_blackman_4(const int n, const int n_total);
double fft_window_exponential(const int n, const int n_total);

fftw_plan fft_plan_rspace_to_kspace(std::complex<double> * rspace, std::complex<double> * kspace, const std::array<int,3>& kspace_size, const int & motif_size);
void apply_kspace_phase_factors(jams::MultiArray<std::complex<double>, 5> &kspace);

#endif  // JAMS_MONITOR_FFT_H

