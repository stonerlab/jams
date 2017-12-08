#include <cmath>
#include <cassert>

#include <fftw3.h>

#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/fft.h"
#include "jams/helpers/consts.h"

using std::complex;
using std::vector;

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

fftw_plan fft_plan_rspace_to_kspace(double * rspace, std::complex<double> * kspace, const Vec3i& kspace_size, const int & motif_size) {
  assert(rspace != nullptr);
  assert(kspace != nullptr);
  assert(sum(kspace_size) > 0);

  int rank            = 3;
  int stride          = 3 * motif_size;
  int dist            = 1;
  int num_transforms  = 3 * motif_size;
  int transform_size[3]  = {kspace_size[0], kspace_size[1], kspace_size[2]};

  int * nembed = nullptr;

  return fftw_plan_many_dft_r2c(
          rank,                    // dimensionality
          transform_size, // array of sizes of each dimension
          num_transforms,          // number of transforms
          rspace,        // input: real data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          reinterpret_cast<fftw_complex*>(kspace),        // output: complex data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          FFTW_PATIENT | FFTW_PRESERVE_INPUT);
}

// Precalculates the phase factors within the brilluoin zone and returns then as array
void precalculate_kspace_phase_factors(
        const Vec3i &kspace_size,
        const Vec3 &r_cart,
        vector<complex<double>> &phase_x,
        vector<complex<double>> &phase_y,
        vector<complex<double>> &phase_z) {

  complex<double> two_pi_i_dr;
  complex<double> exp_phase_0;

  phase_x.resize(kspace_size[0]);
  phase_x.resize(kspace_size[1]);
  phase_x.resize(kspace_size[2]);

  two_pi_i_dr = kImagTwoPi* r_cart[0];
  exp_phase_0 = exp(two_pi_i_dr);
  phase_x[0] = exp(-two_pi_i_dr*double((lattice->kspace_size()[0] - 1)));
  for (auto i = 1; i < lattice->kspace_size()[0]; ++i) {
    phase_x[i] = phase_x[i-1]*exp_phase_0;
  }

  two_pi_i_dr = kImagTwoPi* r_cart[1];
  exp_phase_0 = exp(two_pi_i_dr);
  phase_y[0] = exp(-two_pi_i_dr*double((lattice->kspace_size()[1] - 1)));
  for (auto i = 1; i < lattice->kspace_size()[1]; ++i) {
    phase_y[i] = phase_y[i-1]*exp_phase_0;
  }

  two_pi_i_dr = kImagTwoPi* r_cart[2];
  exp_phase_0 = exp(two_pi_i_dr);
  phase_z[0] = exp(-two_pi_i_dr*double((lattice->kspace_size()[2] - 1)));
  for (auto i = 1; i < lattice->kspace_size()[2]; ++i) {
    phase_z[i] = phase_z[i-1]*exp_phase_0;
  }

}

void apply_kspace_phase_factors(jblib::Array<std::complex<double>, 5> &kspace) {
  using namespace globals;

  vector<complex<double>> exp_phase_x(lattice->kspace_size()[0]);
  vector<complex<double>> exp_phase_y(lattice->kspace_size()[1]);
  vector<complex<double>> exp_phase_z(lattice->kspace_size()[2]);

  for (auto m = 0; m < lattice->motif_size(); ++m) {
    // calculate phase factors
    auto r_cart = lattice->fractional_to_cartesian(lattice->motif_atom(m).pos);

    precalculate_kspace_phase_factors(lattice->kspace_size(), r_cart, exp_phase_x, exp_phase_y, exp_phase_z);

    // normalize the transform
    for (auto i = 0; i < kspace.size(0); ++i) {
      for (auto j = 0; j < kspace.size(1); ++j) {
        for (auto k = 0; k < kspace.size(2); ++k) {
          for (auto n = 0; n < 3; ++n) {
            kspace(i, j, k, m, n) = kspace(i, j, k, m, n) * exp_phase_x[i] * exp_phase_y[j] * exp_phase_z[k];
          }
        }
      }
    }
  }
}

