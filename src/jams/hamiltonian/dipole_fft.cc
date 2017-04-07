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

#include "jblib/containers/array.h"
#include "jblib/containers/matrix.h"

#include "jams/hamiltonian/dipole_fft.h"

using std::pow;
using std::abs;
using std::min;

DipoleHamiltonianFFT::DipoleHamiltonianFFT(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size),
  s_old_(globals::s) {
    printf("  FFT Method\n");

    if (::lattice->num_materials() > 1) {
        jams_error("DipoleHamiltonianFFT only supports single species calculations at the moment");
    }


    double r_abs;
    jblib::Vec3<double> r_ij, eij, s_j;
    jblib::Vec3<int> pos;
    jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

    jblib::Vec3<int> L_max(0, 0, 0);
    jblib::Vec3<double> super_cell_dim(0.0, 0.0, 0.0);

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice->size(n));
    }

    r_cutoff_ = settings["r_cutoff"];

    printf("    r_cutoff: %f\n", r_cutoff_);

    k_cutoff_ = 100000;
    if (settings.exists("k_cutoff")) {
        k_cutoff_ = settings["k_cutoff"];
    }
    printf("    k_cutoff: %d\n", k_cutoff_);

    const double prefactor = kVacuumPermeadbility*kBohrMagneton/(4*kPi*pow(::lattice->parameter(),3));

    for (int n = 0; n < 3; ++n) {
        kspace_size_[n] = ::lattice->num_unit_cells(n);
    }

    kspace_padded_size_ = kspace_size_;

    for (int n = 0; n < 3; ++n) {
        if (!::lattice->is_periodic(n)) {
            kspace_padded_size_[n] = 2*::lattice->num_unit_cells(n);
        }
    }

    jblib::Array<double, 5> wij(kspace_padded_size_[0],
                                kspace_padded_size_[1],
                                kspace_padded_size_[2],
                                3,
                                3);

    std::fill(wij.data(), wij.data()+wij.elements(), 0.0);

    for (int i = 0; i < kspace_padded_size_.x; ++i) {
        for (int j = 0; j < kspace_padded_size_.y; ++j) {
            for (int k = 0; k < kspace_padded_size_.z; ++k) {
                if (unlikely(i == 0 && j == 0 && k == 0)) continue;

                pos = jblib::Vec3<int>(i, j, k);

                for (int n = 0; n < 3; ++n) {
                    if (pos[n] > kspace_padded_size_[n]/2) {
                        pos[n] = periodic_shift(pos[n], kspace_padded_size_[n]/2) - kspace_padded_size_[n]/2;
                    }
                }
                r_ij = jblib::Vec3<double>(pos.x, pos.y, pos.z);
                r_ij  = ::lattice->fractional_to_cartesian(r_ij);

                // r_ij = ::lattice->minimum_image(jblib::Vec3<double>(0, 0, 0), ::lattice->fractional_to_cartesian(pos));
                r_abs = abs(r_ij);

                if (r_abs > r_cutoff_ || unlikely(r_abs < 1e-5)) continue;

                eij = r_ij / r_abs;

                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        // divide by product(kspace_padded_size_) instead or normalizing the FFT result later
                        wij(i, j, k, m, n) += globals::mus(0)*prefactor*(3*eij[m]*eij[n] - Id[m][n])/double(pow(r_abs, 3)*product(kspace_padded_size_));
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------------





    h_nonlocal_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    s_nonlocal_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    s_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    h_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    w_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3, 3);

    for (int i = 0, iend = h_recip_.elements(); i < iend; ++i) {
        h_recip_[i][0] = 0.0;
        h_recip_[i][1] = 0.0;
    }

    printf("    kspace size: %d %d %d\n", kspace_size_[0], kspace_size_[1], kspace_size_[2]);
    printf("    kspace padded size: %d %d %d\n", kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]);

    printf("    planning FFTs\n");

    int rank = 3;
    int stride = 3;
    int dist = 1;
    int num_transforms = 3;
    int * nembed = NULL;

    spin_fft_forward_transform_
        = fftw_plan_many_dft_r2c(rank, &kspace_padded_size_[0], num_transforms,
                                 s_nonlocal_.data(),  nembed, stride, dist,
                                 s_recip_.data(), nembed, stride, dist,
                                 FFTW_PATIENT|FFTW_PRESERVE_INPUT);

    field_fft_backward_transform_
        = fftw_plan_many_dft_c2r(rank, &kspace_padded_size_[0], num_transforms,
                                 h_recip_.data(), nembed, stride, dist,
                                 h_nonlocal_.data(),  nembed, stride, dist,
                                 FFTW_PATIENT);

    rank = 3;
    stride = 9;
    dist = 1;
    num_transforms = 9;

    interaction_fft_transform_
        = fftw_plan_many_dft_r2c(rank, &kspace_padded_size_[0], num_transforms,
            wij.data(),  nembed, stride, dist,
            w_recip_.data(), nembed, stride, dist,
            FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

    printf("    transform interaction matrix\n");

    fftw_execute(interaction_fft_transform_);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianFFT::calculate_total_energy() {
    double e_total = 0.0;
    jblib::Vec3<int> pos;

    calculate_nonlocal_field();
    for (int i = 0; i < globals::num_spins; ++i) {
        pos = ::lattice->super_cell_pos(i);
       e_total += -(globals::s(i,0)*h_nonlocal_(pos.x, pos.y, pos.z, 0)
                     + globals::s(i,1)*h_nonlocal_(pos.x, pos.y, pos.z, 1)
                     + globals::s(i,2)*h_nonlocal_(pos.x, pos.y, pos.z, 2))*globals::mus(i);
    }

    return e_total;
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
        h[m] += h_nonlocal_(pos.x, pos.y, pos.z, m);
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
        h[m] += h_nonlocal_(pos.x, pos.y, pos.z, m);
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

    if (std::equal(&s_old_[0], &s_old_[0]+globals::num_spins3, &globals::s[0]) && !first_run) {
        return;
    } else {
        s_old_ = globals::s;
    }

    for (int i = 0, iend = s_recip_.elements(); i < iend; ++i) {
        s_nonlocal_[i];
    }

    for (int i = 0; i < globals::num_spins; ++i) {
         pos = ::lattice->super_cell_pos(i);
         for (int m = 0; m < 3; ++m) {
            s_nonlocal_(pos.x, pos.y, pos.z, m) = globals::s(i, m);
        }
    }

    for (int i = 0, iend = h_recip_.elements(); i < iend; ++i) {
        h_recip_[i][0] = 0.0;
        h_recip_[i][1] = 0.0;
    }

    fftw_execute(spin_fft_forward_transform_);

    // perform convolution as multiplication in fourier space

    for (i = 0, iend = min(kspace_padded_size_[0], k_cutoff_); i < iend; ++i) {
      for (j = 0, jend = min(kspace_padded_size_[1], k_cutoff_); j < jend; ++j) {
        for (k = 0, kend = min(kspace_padded_size_[2]/2+1, k_cutoff_); k < kend; ++k) {
            if (i*i + j*j + k*k < k_cutoff_*k_cutoff_) {
                for(m = 0; m < 3; ++m) {
                    h_recip_(i,j,k,m)[0] = ( w_recip_(i,j,k,m,0)[0]*s_recip_(i,j,k,0)[0]-w_recip_(i,j,k,m,0)[1]*s_recip_(i,j,k,0)[1] )
                                         + ( w_recip_(i,j,k,m,1)[0]*s_recip_(i,j,k,1)[0]-w_recip_(i,j,k,m,1)[1]*s_recip_(i,j,k,1)[1] )
                                         + ( w_recip_(i,j,k,m,2)[0]*s_recip_(i,j,k,2)[0]-w_recip_(i,j,k,m,2)[1]*s_recip_(i,j,k,2)[1] );

                    h_recip_(i,j,k,m)[1] = ( w_recip_(i,j,k,m,0)[0]*s_recip_(i,j,k,0)[1]+w_recip_(i,j,k,m,0)[1]*s_recip_(i,j,k,0)[0] )
                                         + ( w_recip_(i,j,k,m,1)[0]*s_recip_(i,j,k,1)[1]+w_recip_(i,j,k,m,1)[1]*s_recip_(i,j,k,1)[0] )
                                         + ( w_recip_(i,j,k,m,2)[0]*s_recip_(i,j,k,2)[1]+w_recip_(i,j,k,m,2)[1]*s_recip_(i,j,k,2)[0] );
                }
            }
        }
      }
    }

    fftw_execute(field_fft_backward_transform_);

    first_run = false;
}

// --------------------------------------------------------------------------