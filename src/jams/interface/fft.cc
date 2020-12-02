#include <cmath>
#include <cassert>

#include "jams/core/types.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/interface/fft.h"
#include "jams/helpers/consts.h"
#include "jams/interface/fft.h"

using std::complex;
using std::vector;

double fft_window_default(const int n, const int n_total) {
  return fft_window_blackman_4(n, n_total);
} 

double fft_window_hann(const int n, const int n_total) {
  return 0.50 - 0.50*cos((kTwoPi*n)/double(n_total-1));
}

double fft_window_blackman_4(const int n, const int n_total) {
  // F. J. Harris, Proc. IEEE 66, 51 (1978)
  const double a0 = 0.40217, a1 = 0.49704, a2 = 0.09392, a3 = 0.00183;
  const double x = (kTwoPi * n)/double(n_total-1);
  return a0 - a1 * cos(x) + a2 * cos(2 * x) - a3 * cos(3 * x);
}

double fft_window_exponential(const int n, const int n_total) {
  const double tau = 0.5 * n_total * (8.69  / 60.0);
  return exp(-abs(n - 0.5 * (n_total-1)) / tau);
}

double fft_window_hamming(const int n, const int n_total) {
  return (25.0/46.0) - (1.0 - (25.0/46.0))*cos((kTwoPi*n)/double(n_total-1));
}

double fft_window_nuttall(const int n, const int n_total) {
  double a0 = 0.355768, a1 = 0.487396, a2= 0.144232, a3 = 0.012604;
  return a0 - a1 * cos(2*kPi*n / n_total) + a2 * cos(4*kPi*n / n_total) - a3 * cos(6*kPi*n / n_total);
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
  phase_y.resize(kspace_size[1]);
  phase_z.resize(kspace_size[2]);

  two_pi_i_dr = kImagTwoPi * r_cart[0];
  exp_phase_0 = exp(two_pi_i_dr);
  phase_x[0] = exp(-two_pi_i_dr * double(kspace_size[0] - 1));
  for (auto i = 1; i < phase_x.size(); ++i) {
    phase_x[i] = phase_x[i-1] * exp_phase_0;
  }

  two_pi_i_dr = kImagTwoPi * r_cart[1];
  exp_phase_0 = exp(two_pi_i_dr);
  phase_y[0] = exp(-two_pi_i_dr * double(kspace_size[1] - 1));
  for (auto i = 1; i < phase_y.size(); ++i) {
    phase_y[i] = phase_y[i-1] * exp_phase_0;
  }

  two_pi_i_dr = kImagTwoPi * r_cart[2];
  exp_phase_0 = exp(two_pi_i_dr);
  phase_z[0] = exp(-two_pi_i_dr * double(kspace_size[2] - 1));
  for (auto i = 1; i < phase_z.size(); ++i) {
    phase_z[i] = phase_z[i-1] * exp_phase_0;
  }
}

void apply_kspace_phase_factors(jams::MultiArray<std::complex<double>, 5> &kspace) {
  using namespace globals;

  std::vector<complex<double>> exp_phase_x(lattice->kspace_size()[0]);
  std::vector<complex<double>> exp_phase_y(lattice->kspace_size()[1]);
  std::vector<complex<double>> exp_phase_z(lattice->kspace_size()[2]);

  for (auto m = 0; m < lattice->num_motif_atoms(); ++m) {
    auto r_cart = lattice->fractional_to_cartesian(lattice->motif_atom(m).position);

    precalculate_kspace_phase_factors(lattice->kspace_size(), r_cart, exp_phase_x, exp_phase_y, exp_phase_z);

    for (auto i = 0; i < kspace.size(0); ++i) {
      for (auto j = 0; j < kspace.size(1); ++j) {
        for (auto k = 0; k < kspace.size(2); ++k) {
          std::complex<double> phase = exp_phase_x[i] * exp_phase_y[j] * exp_phase_z[k];
          for (auto n = 0; n < 3; ++n) {
            kspace(i, j, k, m, n) = kspace(i, j, k, m, n) * phase;
          }
        }
      }
    }
  }
}

fftw_plan fft_plan_rspace_to_kspace(std::complex<double> * rspace, std::complex<double> * kspace, const Vec3i& kspace_size, const int & motif_size) {
  assert(rspace != nullptr);
  assert(kspace != nullptr);
  assert(sum(kspace_size) > 0);

  int rank            = 3;
  int stride          = 3 * motif_size;
  int dist            = 1;
  int num_transforms  = 3 * motif_size;
  int transform_size[3]  = {kspace_size[0], kspace_size[1], kspace_size[2]};

  int * nembed = nullptr;

  return fftw_plan_many_dft(
          rank,                    // dimensionality
          transform_size, // array of sizes of each dimension
          num_transforms,          // number of transforms
          reinterpret_cast<fftw_complex*>(rspace),        // input: real data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          jams::fftw::complex_cast(kspace),        // output: complex data
          nembed,                  // number of embedded dimensions
          stride,                  // memory stride between elements of one fft dataset
          dist,                    // memory distance between fft datasets
          FFTW_FORWARD,
          FFTW_PATIENT | FFTW_PRESERVE_INPUT);
}

jams::MultiArray<double, 5> fft_zero_pad_kspace(const jams::MultiArray<double, 2>& rspace_data, const Vec3i& kspace_size, const Vec3i& kspace_padded_size, const int & num_sites) {
  jams::MultiArray<double, 5> padded_rspace_data(kspace_padded_size[0], kspace_padded_size[1], kspace_padded_size[2],
                                                 num_sites, 3);
  zero(padded_rspace_data);

  for (auto i = 0; i < kspace_size[0]; ++i) {
    for (auto j = 0; j < kspace_size[1]; ++j) {
      for (auto k = 0; k < kspace_size[2]; ++k) {
        for (auto m = 0; m < num_sites; ++m) {
          int index = ((i * kspace_size[1] + j) * kspace_size[2] + k) * num_sites + m;
          for (auto n : {0, 1, 2}) {
            padded_rspace_data(i, j, k, m, n) = rspace_data(index, n);
          }
        }
      }
    }
  }

  return padded_rspace_data;
}

void fft_supercell_vector_field_to_kspace(const jams::MultiArray<double, 2>& rspace_data, jams::MultiArray<Vec3cx,4>& kspace_data,  const Vec3i& kspace_size, const Vec3i& kspace_padded_size, const int & num_sites) {
  assert(rspace_data.elements() == 3 * num_sites * product(kspace_size));

  kspace_data.resize(kspace_padded_size[0], kspace_padded_size[1], kspace_padded_size[2]/2 + 1, num_sites);

  auto fourier_transform = [&](const double* rspace_data_ptr) {
      int rank              = 3;
      int transform_size[3] = {kspace_padded_size[0], kspace_padded_size[1], kspace_padded_size[2]};
      int num_transforms    = 3 * num_sites;
      int *nembed           = nullptr;
      int stride            = 3 * num_sites;
      int dist              = 1;

      // FFTW_PRESERVE_INPUT is not supported for r2c arrays but FFTW_ESTIMATE doe not overwrite
      auto plan = fftw_plan_many_dft_r2c(
          rank, transform_size, num_transforms,
          const_cast<double*>(rspace_data_ptr), nembed, stride, dist,
          jams::fftw::complex_cast(kspace_data.data()), nembed, stride, dist,
          FFTW_ESTIMATE);

      assert(plan);
      fftw_execute(plan);
      fftw_destroy_plan(plan);
      element_scale(kspace_data, 1.0/sqrt(product(kspace_size)));
  };

  if (kspace_size == kspace_padded_size) {
    fourier_transform(rspace_data.data());
  } else {
    auto rspace_padded_data = fft_zero_pad_kspace(rspace_data, kspace_size, kspace_padded_size, num_sites);
    fourier_transform(rspace_padded_data.data());
  }
}

void fft_supercell_scalar_field_to_kspace(const jams::MultiArray<double, 1>& rspace_data, jams::MultiArray<Complex,4>& kspace_data, const Vec3i& kspace_size, const int & num_sites) {
  assert(rspace_data.elements() == product(kspace_size));

  // assuming this is not a costly operation because .resize() already checks if it is the same size
  kspace_data.resize(kspace_size[0], kspace_size[1], kspace_size[2]/2 + 1, num_sites);

  int rank              = 3;
  int transform_size[3] = {kspace_size[0], kspace_size[1], kspace_size[2]};
  int num_transforms    = num_sites;
  int *nembed           = nullptr;
  int stride            = num_sites;
  int dist              = 1;

  // FFTW_PRESERVE_INPUT is not supported for r2c arrays but FFTW_ESTIMATE doe not overwrite
  auto plan = fftw_plan_many_dft_r2c(
      rank, transform_size, num_transforms,
      const_cast<double*>(rspace_data.begin()), nembed, stride, dist,
      jams::fftw::complex_cast(kspace_data.begin()), nembed, stride, dist,
      FFTW_ESTIMATE);

  assert(plan);
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  element_scale(kspace_data, 1.0/sqrt(product(lattice->kspace_size())));
}
