#include <cmath>

#include "fft.h"
#include "consts.h"

double fft_window_default(const int n, const int n_total) {
  return fft_window_blackman_4(n, n_total);
} 

double fft_window_hanning(const int n, const int n_total) {
  return 0.50 - 0.50*cos((kTwoPi*n)/double(n_total));
}

double fft_window_blackman_4(const int n, const int n_total) {
  // F. J. Harris, Proc. IEEE 66, 51 (1978)
  const double a0 = 0.40217, a1 = 0.49704, a2 = 0.09392, a3 = 0.00183;
  const double x = (kTwoPi * n)/double(n_total);
  return a0 - a1 * cos(x) + a2 * cos(2 * x) - a3 * cos(3 * x);
}
