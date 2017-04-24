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

//---------------------------------------------------------------------

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

//---------------------------------------------------------------------

DipoleHamiltonianFFT::DipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size),
  r_cutoff_(0),
  k_cutoff_(0),
  h_(globals::num_spins, 3),
  rspace_s_(),
  rspace_h_(),
  kspace_size_(0, 0, 0),
  kspace_padded_size_(0, 0, 0),
  kspace_s_(),
  kspace_h_(),
  kspace_tensors_(),
  fft_s_rspace_to_kspace(nullptr),
  fft_h_kspace_to_rspace(nullptr)
{
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

    rspace_s_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    rspace_h_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    kspace_s_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    kspace_h_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);

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
            rank,                    // dimensionality
            &kspace_padded_size_[0], // array of sizes of each dimension
            num_transforms,          // number of transforms
            rspace_s_.data(),        // input: real data
            nembed,                  // number of embedded dimensions 
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            kspace_s_.data(),        // output: complex data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT);

    fft_h_kspace_to_rspace
        = fftw_plan_many_dft_c2r(
            rank,                    // dimensionality
            &kspace_padded_size_[0], // array of sizes of each dimension
            num_transforms,          // number of transforms
            kspace_h_.data(),        // input: complex data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset 
            dist,                    // memory distance between fft datasets
            rspace_h_.data(),        // output: real data
            nembed,                  // number of embedded dimensions
            stride,                  // memory stride between elements of one fft dataset
            dist,                    // memory distance between fft datasets
            FFTW_PATIENT);

    kspace_tensors_.resize(lattice->num_unit_cell_positions());

    for (int pos_i = 0; pos_i < lattice->num_unit_cell_positions(); ++pos_i) {
        for (int pos_j = 0; pos_j < lattice->num_unit_cell_positions(); ++pos_j) {
            kspace_tensors_[pos_i].push_back(generate_kspace_dipole_tensor(pos_i, pos_j));
        }
    }
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_total_energy() {
    double e_total = 0.0;

    calculate_fields(h_);
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += (  globals::s(i,0)*h_(i, 0)
                    + globals::s(i,1)*h_(i, 1)
                    + globals::s(i,2)*h_(i, 2) )*globals::mus(i);
    }

    return -0.5*e_total;
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0] * h[0] + s_i[1] * h[1] + s_i[2] * h[2]) * globals::mus(i);
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

//---------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_one_spin_energy_difference(
    const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final)
{
    jblib::Vec3<int> pos;

    calculate_fields(h_);

    return -( (spin_final[0] * h_(i,0) + spin_final[1] * h_(i,1) + spin_final[2] * h_(i,2))
          - (spin_initial[0] * h_(i,0) + spin_initial[1] * h_(i,1) + spin_initial[2] * h_(i,2))) * globals::mus(i);
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.elements() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_one_spin_field(const int i, double h[3]) {
    calculate_fields(h_);

    for (int m = 0; m < 3; ++m) {
        h[m] = h_(i, m);
    }
}

//---------------------------------------------------------------------

void DipoleHamiltonianFFT::calculate_fields(jblib::Array<double, 2>& fields) {
    using std::min;
    using std::pow;

    fields.zero();

    for (int pos_i = 0; pos_i < lattice->num_unit_cell_positions(); ++pos_i) {

        kspace_h_.zero();

        for (int pos_j = 0; pos_j < lattice->num_unit_cell_positions(); ++pos_j) {

            const double mus_j = lattice->unit_cell_material(pos_j).moment;

            rspace_s_.zero();

            for (int kx = 0; kx < kspace_size_[0]; ++kx) {
                for (int ky = 0; ky < kspace_size_[1]; ++ky) {
                    for (int kz = 0; kz < kspace_size_[2]; ++kz) {
                        const int index = lattice->site_index_by_unit_cell(kx, ky, kz, pos_j);
                        for (int m = 0; m < 3; ++m) {
                            rspace_s_(kx, ky, kz, m) = globals::s(index, m);
                        }
                    }
                }
            }

            fftw_execute(fft_s_rspace_to_kspace);

            // perform convolution as multiplication in fourier space
            for (int i = 0; i < kspace_padded_size_[0]; ++i) {
                for (int j = 0; j < kspace_padded_size_[1]; ++j) {
                    for (int k = 0; k < (kspace_padded_size_[2]/2)+1; ++k) {
                    const Vec3 q(i,j,k);

                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                std::complex<double> wq(
                                    kspace_tensors_[pos_i][pos_j](i,j,k,m,n)[0], 
                                    kspace_tensors_[pos_i][pos_j](i,j,k,m,n)[1]);

                                std::complex<double> sq(kspace_s_(i,j,k,n)[0], kspace_s_(i,j,k,n)[1]);

                                std::complex<double> hq = wq * sq;

                                kspace_h_(i,j,k,m)[0] += mus_j * hq.real();
                                kspace_h_(i,j,k,m)[1] += mus_j * hq.imag(); 
                            }   
                        }
                    }
                }
            }
        }  // unit cell pos_j

        rspace_h_.zero();
        fftw_execute(fft_h_kspace_to_rspace);

        for (int i = 0; i < kspace_size_[0]; ++i) {
            for (int j = 0; j < kspace_size_[1]; ++j) {
                for (int k = 0; k < kspace_size_[2]; ++k) {
                    const int index = lattice->site_index_by_unit_cell(i, j, k, pos_i);
                    for (int m = 0; m < 3; ++ m) {
                        fields(index, m) += rspace_h_(i, j, k, m);
                    }   
                }
            }
        }
    
    }  // unit cell pos_i
}

//---------------------------------------------------------------------

jblib::Array<fftw_complex, 5> 
DipoleHamiltonianFFT::generate_kspace_dipole_tensor(const int pos_i, const int pos_j) {
    using std::pow;

    const Vec3 r_frac_i = lattice->unit_cell_position(pos_i);
    const Vec3 r_frac_j = lattice->unit_cell_position(pos_j);

    const Vec3 r_cart_i = lattice->unit_cell_position_cart(pos_i);
    const Vec3 r_cart_j = lattice->unit_cell_position_cart(pos_j);

    jblib::Array<double, 5> rspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2],
        3, 3);

    jblib::Array<fftw_complex, 5> kspace_tensor(
        kspace_padded_size_[0],
        kspace_padded_size_[1],
        kspace_padded_size_[2],
        3, 3);


    rspace_tensor.zero();
    kspace_tensor.zero();

    const double fft_normalization_factor = 1.0 / product(kspace_padded_size_);
    const double v = pow(lattice->parameter(), 3);
    const double w0 = fft_normalization_factor * kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * v);


    for (int kx = 0; kx < kspace_size_[0]; ++kx) {
        for (int ky = 0; ky < kspace_size_[1]; ++ky) {
            for (int kz = 0; kz < kspace_size_[2]; ++kz) {

                if (kx == 0 && ky == 0 && kz == 0 && pos_i == pos_j) {
                    // self interaction on the same sublattice
                    continue;
                } 

                const Vec3 r_ij = 
                    lattice->minimum_image(r_cart_j, 
                        lattice->generate_position(r_frac_i, {kx, ky, kz})); // generate_position requires FRACTIONAL coordinate

                const auto r_abs_sq = r_ij.norm_sq();

                if (r_abs_sq > pow(r_cutoff_, 2)) {
                    // outside of cutoff radius
                    continue;
                }

                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        rspace_tensor(kx, ky, kz, m, n) 
                            = w0 * (3 * r_ij[m] * r_ij[n] - r_abs_sq * Id[m][n]) / pow(sqrt(r_abs_sq), 5);
                    }
                }
            }
        }
    }

    int rank            = 3;
    int stride          = 9;
    int dist            = 1;
    int num_transforms  = 9;
    int * nembed        = NULL;

    fftw_plan fft_dipole_tensor_rspace_to_kspace
        = fftw_plan_many_dft_r2c(
            rank,                       // dimensionality
            &kspace_padded_size_[0],    // array of sizes of each dimension
            num_transforms,             // number of transforms
            rspace_tensor.data(),       // input: real data
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            kspace_tensor.data(),       // output: real dat
            nembed,                     // number of embedded dimensions
            stride,                     // memory stride between elements of one fft dataset
            dist,                       // memory distance between fft datasets
            FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

    fftw_execute(fft_dipole_tensor_rspace_to_kspace);
    fftw_destroy_plan(fft_dipole_tensor_rspace_to_kspace);

    return kspace_tensor;
}

//---------------------------------------------------------------------