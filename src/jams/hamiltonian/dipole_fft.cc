#include <cassert>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <complex>

#include "jams/core/error.h"
#include "jams/core/lattice.h"
#include "jams/core/globals.h"
#include "jams/core/consts.h"
#include "jams/core/utils.h"
#include "jams/core/output.h"

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

#include "jams/hamiltonian/dipole_fft.h"

using std::pow;
using std::abs;
using std::min;

namespace {
    const jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );
}

DipoleHamiltonianFFT::~DipoleHamiltonianFFT() {
    if (fft_s_rspace_to_kspace) {
        fftw_destroy_plan(fft_s_rspace_to_kspace);
        fft_s_rspace_to_kspace = nullptr;
    }

    if (fft_h_kspace_to_rspace) {
        fftw_destroy_plan(fft_h_kspace_to_rspace);
        fft_h_kspace_to_rspace = nullptr;
    }
}

DipoleHamiltonianFFT::DipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size),
  r_cutoff_(0),
  k_cutoff_(0),
  cached_s_(globals::s),
  rspace_s_(),
  rspace_h_(),
  kspace_size_(0, 0, 0),
  kspace_padded_size_(0, 0, 0),
  kspace_s_(),
  kspace_h_(),
  kspace_dipole_tensor_(),
  fft_s_rspace_to_kspace(nullptr),
  fft_h_kspace_to_rspace(nullptr)
{

    if (::lattice->num_materials() > 1) {
        jams_error("DipoleHamiltonianFFT only supports single species calculations at the moment");
    }

    jblib::Vec3<int> L_max(0, 0, 0);

    r_cutoff_ = double(settings["r_cutoff"]);
    output->write("  r_cutoff: %e\n", r_cutoff_);

    k_cutoff_ = 100000;
    if (settings.exists("k_cutoff")) {
        k_cutoff_ = settings["k_cutoff"];
    }
    output->write("  k_cutoff: %e\n", r_cutoff_);

    for (int n = 0; n < 3; ++n) {
        kspace_size_[n] = ::lattice->num_unit_cells(n);
    }

    kspace_padded_size_ = kspace_size_;

    for (int n = 0; n < 3; ++n) {
        if (!::lattice->is_periodic(n)) {
            kspace_padded_size_[n] = kspace_size_[n] * 2;
        }
    }

    //---------------------------------------------------------------------
    // setup realspace dipole tensor
    //---------------------------------------------------------------------

    jblib::Array<double, 5> dipole_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2],
        3, 3);

    dipole_tensor.zero();

    const double dipole_prefactor = 
        kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * std::pow(::lattice->parameter(),3));

    const double fft_normalization_factor = 1.0 / product(kspace_padded_size_);

    const int i = 0;
    for (int j = 0; j < globals::num_spins; ++j) {
        if (j == i) continue;

        auto r_ij = lattice->displacement(i, j);

        const auto r_abs_sq = r_ij.norm_sq();

        if (r_abs_sq > (r_cutoff_*r_cutoff_)) continue;
        
        const auto r_abs = sqrt(r_abs_sq);

        // divide by product(kspace_padded_size_) instead or normalizing the FFT result later
        const auto w0 = globals::mus(i) * dipole_prefactor / std::pow(r_abs, 5);
        
        auto p = ::lattice->super_cell_pos(j);

        assert(p[0] < kspace_padded_size_[0]);
        assert(p[1] < kspace_padded_size_[1]);
        assert(p[2] < kspace_padded_size_[2]);

        for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
                dipole_tensor(p[0], p[1], p[2], m, n) 
                    += w0 * (3.0 * r_ij[m] * r_ij[n] - r_abs_sq * Id[m][n]) * fft_normalization_factor;
            }
        }
    }
    //---------------------------------------------------------------------

    rspace_s_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    rspace_h_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    kspace_s_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    kspace_h_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    kspace_dipole_tensor_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3, 3);

    for (int i = 0, iend = kspace_h_.elements(); i < iend; ++i) {
        kspace_h_[i][0] = 0.0;
        kspace_h_[i][1] = 0.0;
    }

    output->write("    kspace size: %d %d %d\n", kspace_size_[0], kspace_size_[1], kspace_size_[2]);
    output->write("    kspace padded size: %d %d %d\n", kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]);

    output->write("    planning FFTs\n");

    int rank            = 3;           
    int stride          = 3;
    int dist            = 1;
    int num_transforms  = 3;

    int * nembed = NULL;

    fft_s_rspace_to_kspace
        = fftw_plan_many_dft_r2c(
            rank,                       // dimensionality
            &kspace_padded_size_[0],    // array of sizes of each dimension
            num_transforms,             // number of transforms
            rspace_s_.data(),         // input: real data
            nembed,                     // number of embedded dimensions 
            stride,                     // memory stride between elements of one fft dataset 
            dist,                       // memory distance between fft datasets
            kspace_s_.data(),            // output: complex data
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset 
            dist,                       // memory distance between fft datasets
            FFTW_PATIENT|FFTW_PRESERVE_INPUT);

    fft_h_kspace_to_rspace
        = fftw_plan_many_dft_c2r(
            rank,                       // dimensionality
            &kspace_padded_size_[0],    // array of sizes of each dimension
            num_transforms,             // number of transforms
            kspace_h_.data(),            // input: complex data
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset 
            dist,                       // memory distance between fft datasets
            rspace_h_.data(),         // output: real data
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            FFTW_PATIENT);

    rank            = 3;
    stride          = 9;
    dist            = 1;
    num_transforms  = 9;

    fftw_plan fft_dipole_tensor_rspace_to_kspace
        = fftw_plan_many_dft_r2c(
            rank,                       // dimensionality
            &kspace_padded_size_[0],    // array of sizes of each dimension
            num_transforms,             // number of transforms
            dipole_tensor.data(),                 // input: real data
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            kspace_dipole_tensor_.data(),            // output: real dat
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);

    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_total_energy() {
    double e_total = 0.0;

    calculate_nonlocal_field();
    for (int i = 0; i < globals::num_spins; ++i) {
        auto p = ::lattice->super_cell_pos(i);
        e_total += (   globals::s(i,0)*rspace_h_(p.x, p.y, p.z, 0)
                    + globals::s(i,1)*rspace_h_(p.x, p.y, p.z, 1)
                    + globals::s(i,2)*rspace_h_(p.x, p.y, p.z, 2))*globals::mus(i);
    }

    return -0.5*e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2])*globals::mus(i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy_difference(
    const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final)
{
    jblib::Vec3<int> pos;

    double h[3] = {0, 0, 0};

    calculate_nonlocal_field();
    for (int m = 0; m < 3; ++m) {
        pos = ::lattice->super_cell_pos(i);
        h[m] += rspace_h_(pos.x, pos.y, pos.z, m);
    }

    return -( (spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2])
          - (spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]))*globals::mus(i);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.elements() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_one_spin_field(const int i, double h[3]) {
    jblib::Vec3<int> pos;

    for (int m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }

    calculate_nonlocal_field();
    for (int m = 0; m < 3; ++m) {
        pos = ::lattice->super_cell_pos(i);
        h[m] += rspace_h_(pos.x, pos.y, pos.z, m);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_fields(jblib::Array<double, 2>& energies) {

}

void DipoleHamiltonianFFT::calculate_nonlocal_field() {
    using std::min;
    static bool first_run = true;

    int i, iend, j, jend, k, kend, m;
    jblib::Vec3<int> pos;

    if (std::equal(&cached_s_[0], &cached_s_[0]+globals::num_spins3, &globals::s[0]) && !first_run) {
        return;
    } else {
        cached_s_ = globals::s;
    }

    rspace_s_.zero();

    for (int i = 0; i < globals::num_spins; ++i) {
         pos = ::lattice->super_cell_pos(i);
         for (int m = 0; m < 3; ++m) {
            rspace_s_(pos.x, pos.y, pos.z, m) = globals::s(i, m);
        }
    }

    for (int i = 0, iend = kspace_h_.elements(); i < iend; ++i) {
        kspace_h_[i][0] = 0.0;
        kspace_h_[i][1] = 0.0;
    }

    fftw_execute(fft_s_rspace_to_kspace);

    // perform convolution as multiplication in fourier space
    for (i = 0; i < kspace_padded_size_[0]; ++i) {
      for (j = 0; j < kspace_padded_size_[1]; ++j) {
        for (k = 0; k < (kspace_padded_size_[2]/2)+1; ++k) {
          for (int m = 0; m < 3; ++ m) {
            for (int n = 0; n < 3; ++ n) {
              kspace_h_(i,j,k,m)[0] += kspace_dipole_tensor_(i,j,k,m,n)[0]*kspace_s_(i,j,k,n)[0]-kspace_dipole_tensor_(i,j,k,m,n)[1]*kspace_s_(i,j,k,n)[1];
              kspace_h_(i,j,k,m)[1] += kspace_dipole_tensor_(i,j,k,m,n)[0]*kspace_s_(i,j,k,n)[1]+kspace_dipole_tensor_(i,j,k,m,n)[1]*kspace_s_(i,j,k,n)[0];
            }   
          }
        }
      }
    }

    fftw_execute(fft_h_kspace_to_rspace);

    first_run = false;
}

// --------------------------------------------------------------------------
