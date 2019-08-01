// Copyright 2016 Joseph Barker. All rights reserved.

// Always include this file for FFT/FFTW usage. Do not include fftw3.h directly into any other file.

#ifndef JAMS_MONITOR_FFT_H
#define JAMS_MONITOR_FFT_H

// This hack avoids an incompatibility between CUDA and versions of fftw and mkl which do not detect __CUDACC__ as the
// compiler, resulting in an compile error that __float128 is not defined. We never use the quad precision routines
// in JAMS, so by undefining __GNUC__, the FFTW header will ignore the quad routines.
#define ORIG_GNUC __GNUC__
#undef __GNUC__
#include <fftw3.h>
#define __GNUC__ ORIG_GNUC

#include <complex>
#include <array>

#include "jams/containers/multiarray.h"

using Complex = std::complex<double>;

namespace jams {
    struct PeriodogramProps {
        int length    = 1000;
        int overlap    = 500;
        double sample_time  = 1.0;
    };
}

#define FFTW_COMPLEX_CAST(x) reinterpret_cast<fftw_complex*>(x)

// Windowing functions
double fft_window_default(const int n, const int n_total);
double fft_window_hann(const int n, const int n_total);
double fft_window_hamming(const int n, const int n_total);
double fft_window_blackman_4(const int n, const int n_total);
double fft_window_exponential(const int n, const int n_total);
double fft_window_nuttall(const int n, const int n_total);

fftw_plan fft_plan_rspace_to_kspace(std::complex<double> * rspace, std::complex<double> * kspace, const std::array<int,3>& kspace_size, const int & motif_size);
void apply_kspace_phase_factors(jams::MultiArray<std::complex<double>, 5> &kspace);

void fft_supercell_vector_field_to_kspace(const jams::MultiArray<double, 2>& rspace_data, jams::MultiArray<Vec3cx,4>& kspace_data, const Vec3i& kspace_size, const int & num_sites);
void fft_supercell_scalar_field_to_kspace(const jams::MultiArray<double, 1>& rspace_data, jams::MultiArray<Complex,4>& kspace_data, const Vec3i& kspace_size, const int & num_sites);

template <std::size_t N>
struct FFTWHermitianIndex {
    Vec<int, N> offset;
    bool conj;
};

// returns true if k corresponds to a (virtual) hermitian element
template <std::size_t N>
inline bool fftw_r2c_need_hermitian_symmetric_index(Vec<int, N> &k, const Vec<int, N> &n) {
  return (k[N-1] % n[N-1] + n[N-1]) % n[N-1] >= n[N-1] / 2 + 1;
}

// Maps an index for an FFTW ordered array into the correct location and conjugation
//
// FFTW real to complex (r2c) transforms only give N/2+1 outputs in the last dimensions
// as all other values can be calculated by Hermitian symmetry. This function maps
// the N-dimensional array position of a set of general indicies (positive, negative, larger than
// the fft_size) to the correct location and sets a bool as to whether the value must
// be conjugated
template <std::size_t N>
inline FFTWHermitianIndex<N> fftw_r2c_index(Vec<int, N> k, const Vec<int, N> &n) {
  if (fftw_r2c_need_hermitian_symmetric_index(k, n)) {
    return FFTWHermitianIndex<N>{(-k % n + n) % n, true};
  }
  return FFTWHermitianIndex<N>{(k % n + n) % n, false};
}

#endif  // JAMS_MONITOR_FFT_H

