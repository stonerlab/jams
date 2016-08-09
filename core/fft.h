// Copyright 2016 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_FFT_H
#define JAMS_MONITOR_FFT_H

// Windowing functions
double fft_window_default(const int n, const int n_total);
double fft_window_hanning(const int n, const int n_total);
double fft_window_blackman_4(const int n, const int n_total);

#endif  // JAMS_MONITOR_FFT_H

