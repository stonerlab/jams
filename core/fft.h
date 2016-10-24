// Copyright 2016 Joseph Barker. All rights reserved.

#ifndef JAMS_MONITOR_FFT_H
#define JAMS_MONITOR_FFT_H

// Windowing functions
inline double fft_window_hanning(const int n, const int n_total) {
  return 0.50 - 0.50*cos((kTwoPi*n)/double(n_total));
}

inline double fft_window_blackman_4(const int n, const int n_total) {
  // F. J. Harris, Proc. IEEE 66, 51 (1978)
  const double a0 = 0.40217, a1 = 0.49704, a2 = 0.09392, a3 = 0.00183;
  return a0 - a1 * cos((kTwoPi * n)/double(n_total)) 
  + a2 * cos((kTwoPi * 2 * n)/double(n_total))
  - a3 * cos((kTwoPi * 3 * n)/double(n_total));
}

#endif  // JAMS_MONITOR_FFT_H

