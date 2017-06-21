// Copyright 2016 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_FFT_H
#define JAMS_MONITOR_FFT_H

// Windowing functions
double fft_window_default(const int n, const int n_total);
double fft_window_hanning(const int n, const int n_total);
double fft_window_blackman_4(const int n, const int n_total);

int fft_k_index(const int i, const int size);
double fft_k_point(const int i, const int size);

#endif  // JAMS_MONITOR_FFT_H

