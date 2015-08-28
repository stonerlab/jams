#include "core/globals.h"
#include "core/consts.h"
#include "core/utils.h"

#include "hamiltonian/dipole_Ewald.h"

using std::pow;
using std::min;

DipoleHamiltonianEwald::DipoleHamiltonianEwald(const libconfig::Setting &settings)
: HamiltonianStrategy(settings) {

    ::output.write("  Ewald Method\n");

    if (::lattice.num_materials() > 1) {
        jams_error("DipoleHamiltonianEwald only supports single species calculations at the moment");
    }
    local_interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);

    r_cutoff_ = 3.0;
    k_cutoff_ = 6;
    sigma_ = 2.0*kPi/r_cutoff_;


    ::output.write("    sigma: %f\n", sigma_);
    ::output.write("    r_cutoff: %f\n", r_cutoff_);
    ::output.write("    k_cutoff: %d\n", k_cutoff_);


    double r_abs;
    jblib::Vec3<double> r_ij, eij, s_j;
    jblib::Vec3<int> pos;
    jblib::Matrix<double, 3, 3> tensor;
    jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

    local_interaction_matrix_.resize(globals::num_spins*3, globals::num_spins*3);

    const double prefactor = kVacuumPermeadbility_FourPi*kBohrMagneton/pow(::lattice.parameter(),3);

    // --------------------------------------------------------------------------
    // local real space interactions
    // --------------------------------------------------------------------------

    int local_interaction_counter = 0;
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < globals::num_spins; ++j) {
            if (unlikely(j == i)) continue;
            r_ij  = lattice.position(j) - lattice.position(i);
            r_abs = abs(r_ij);


            if (r_abs > r_cutoff_) continue;

            eij = r_ij / r_abs;
            // TODO: enforce minimum image convention
            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    tensor[m][n] = ((3*eij[m]*eij[n] - Id[m][n])*fG(r_abs, sigma_)/(pow(r_abs, 3)))
                                 - (eij[m]*eij[n]*sqrt(2.0/kPi)*exp(-0.5*pow(r_abs/sigma_, 2))/(pow(r_abs, 3)));
                    // tensor[m][n] = (3*erfc(sigma_*r_abs) + (2*sigma_*r_abs/sqrt(kPi))*(3+2*pow(sigma_*r_abs,2))*exp(-pow(sigma_*r_abs,2))/pow(r_abs,5))*r_ij[m]*r_ij[n]
                    //              - (erfc(sigma_*r_abs) + (2*sigma_*r_abs/sqrt(kPi))*exp(-pow(sigma_*r_abs,2))/pow(r_abs,3));
                }
            }

            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    local_interaction_matrix_.insertValue(3*i + m, 3*j + n, tensor[m][n]*prefactor*globals::mus(j));
                }
            }
            local_interaction_counter++;
        }
    }

    ::output.write("    total local interactions: %d (%d per spin)\n", local_interaction_counter, local_interaction_counter/globals::num_spins);
    local_interaction_matrix_.convertMAP2CSR();
    ::output.write("    local dipole tensor memory (CSR): %3.2f MB\n", local_interaction_matrix_.calculateMemory());

    // std::cerr << "\n\n";

    // --------------------------------------------------------------------------
    // nonlocal reciprocal space interactions
    // --------------------------------------------------------------------------

    for (int n = 0; n < 3; ++n) {
        kspace_size_[n] = ::lattice.num_unit_cells(n);
    }

    kspace_padded_size_ = kspace_size_;

    for (int n = 0; n < 3; ++n) {
        if (!::lattice.is_periodic(n)) {
            kspace_padded_size_[n] = 2*::lattice.num_unit_cells(n);
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

                // convert to +/- r
                for (int n = 0; n < 3; ++n) {
                    if (pos[n] > kspace_padded_size_[n]/2) {
                        pos[n] = periodic_shift(pos[n], kspace_padded_size_[n]/2) - kspace_padded_size_[n]/2;
                    }
                }

                std::cerr << i << "\t" << j << "\t" << k << "\t" << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;

                r_ij = jblib::Vec3<double>(pos.x, pos.y, pos.z);
                r_ij  = ::lattice.fractional_to_cartesian(r_ij);
                r_abs = abs(r_ij);

                eij = r_ij / r_abs;

                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        // divide by product(kspace_padded_size_) instead or normalizing the FFT result later
                        // wij(i, j, k, m, n) += globals::mus(0)*prefactor*(2*eij[m]*eij[n]*gaussian(r_abs, sigma_)/(product(kspace_padded_size_)));
                        wij(i, j, k, m, n) += globals::mus(0)*prefactor*(eij[m]*eij[n]*sqrt(2.0/kPi)*exp(-0.5*pow(r_abs/sigma_, 2))/(product(kspace_padded_size_)*pow(r_abs, 3)));
                    }
                }
            }
        }
    }


    h_nonlocal_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    s_nonlocal_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    s_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    h_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    w_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3, 3);

    for (int i = 0, iend = h_recip_.elements(); i < iend; ++i) {
        h_recip_[i][0] = 0.0;
        h_recip_[i][1] = 0.0;
    }

    ::output.write("    kspace size: %d %d %d\n", kspace_size_[0], kspace_size_[1], kspace_size_[2]);
    ::output.write("    kspace padded size: %d %d %d\n", kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2]);

    ::output.write("    planning FFTs\n");

    spin_fft_forward_transform_   = fftw_plan_many_dft_r2c(3, &kspace_padded_size_[0], 3, s_nonlocal_.data(),  NULL, 3, 1, s_recip_.data(), NULL, 3, 1, FFTW_PATIENT|FFTW_PRESERVE_INPUT);
    field_fft_backward_transform_ = fftw_plan_many_dft_c2r(3, &kspace_padded_size_[0], 3, h_recip_.data(), NULL, 3, 1, h_nonlocal_.data(),  NULL, 3, 1, FFTW_PATIENT);
    interaction_fft_transform_    = fftw_plan_many_dft_r2c(3, &kspace_padded_size_[0], 9, wij.data(),  NULL, 9, 1, w_recip_.data(), NULL, 9, 1, FFTW_ESTIMATE|FFTW_PRESERVE_INPUT);

    ::output.write("    transform interaction matrix\n");

    fftw_execute(interaction_fft_transform_);

//     for (int i = 0; i < kspace_padded_size_.x; ++i) {
//         for (int j = 0; j < kspace_padded_size_.y; ++j) {
//             for (int k = 0; k < kspace_padded_size_.z/2 + 1; ++k) {
//                 pos = jblib::Vec3<int>(i, j, k);

//                 // convert to +/- k
//                 for (int n = 0; n < 3; ++n) {
//                     if (pos[n] > kspace_padded_size_[n]/2 + 1) {
//                         pos [n] = periodic_shift(pos[n], kspace_padded_size_[n]/2) - kspace_padded_size_[n]/2;
//                     }
//                 }

//                 for (int m = 0; m < 3; ++m) {
//                     for (int n = 0; n < 3; ++n) {
//                         if (abs(pos) > k_cutoff_  ){
//                             w_recip_(i,j,k,m,n)[0] = 0.0;
//                             w_recip_(i,j,k,m,n)[1] = 0.0;
//                         }
//                     }
//                 }
//             }
//         }
//     }

}

// --------------------------------------------------------------------------

double DipoleHamiltonianEwald::calculate_total_energy() {
   assert(energies.size(0) == globals::num_spins);
   double e_total = 0.0;
   double h[3];
   jblib::Vec3<int> pos;



   calculate_nonlocal_ewald_field();
   for (int i = 0; i < globals::num_spins; ++i) {
        pos = ::lattice.super_cell_pos(i);
       e_total += -(globals::s(i,0)*h_nonlocal_(pos.x, pos.y, pos.z, 0)
                     + globals::s(i,1)*h_nonlocal_(pos.x, pos.y, pos.z, 1)
                     + globals::s(i,2)*h_nonlocal_(pos.x, pos.y, pos.z, 2))*globals::mus(i);
   }


   for (int i = 0; i < globals::num_spins; ++i) {
        calculate_local_ewald_field(i, h);
       e_total += -(globals::s(i,0)*h[0] + globals::s(i,1)*h[1] + globals::s(i,2)*h[2])*globals::mus(i);
   }
    return e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianEwald::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2])*globals::mus(i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianEwald::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianEwald::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    jblib::Vec3<int> pos;
    double h[3] = {0.0, 0.0, 0.0};

    calculate_local_ewald_field(i, h);

    calculate_nonlocal_ewald_field();
    for (int m = 0; m < 3; ++m) {
        pos = ::lattice.super_cell_pos(i);
        h[m] += h_nonlocal_(pos.x, pos.y, pos.z, m);
    }

    return ((-(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2])) - (-(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2])))*globals::mus(i);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianEwald::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size(0) == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianEwald::calculate_one_spin_field(const int i, double h[3]) {
    jblib::Vec3<int> pos;

    for (int m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }

    calculate_local_ewald_field(i, h);

    calculate_nonlocal_ewald_field();
    for (int m = 0; m < 3; ++m) {
        pos = ::lattice.super_cell_pos(i);
        h[m] += h_nonlocal_(pos.x, pos.y, pos.z, m);
    }
    // std::cerr << h[0] << "\t" << h[1] << "\t" << h[2] << "\t" << std::endl;
}

// --------------------------------------------------------------------------

void DipoleHamiltonianEwald::calculate_fields(jblib::Array<double, 2>& energies) {

}

void DipoleHamiltonianEwald::calculate_local_ewald_field(const int i, double h[3]) {
    assert(local_interaction_matrix_.getMatrixType() == SPARSE_MATRIX_TYPE_GENERAL);
    for (int m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }
    const double *val = local_interaction_matrix_.valPtr();
    const int    *indx = local_interaction_matrix_.colPtr();
    const int    *ptrb = local_interaction_matrix_.ptrB();
    const int    *ptre = local_interaction_matrix_.ptrE();
    const double *x   = globals::s.data();
    int           k;

    for (int m = 0; m < 3; ++m) {
      int begin = ptrb[3*i+m]; int end = ptre[3*i+m];
      for (int j = begin; j < end; ++j) {
        k = indx[j];
        h[m] = h[m] + x[k]*val[j];
      }
    }
}

void DipoleHamiltonianEwald::calculate_nonlocal_ewald_field() {
    using std::min;
    int i, iend, j, jend, k, kend, m;
    jblib::Vec3<int> pos;


    for (int i = 0, iend = s_recip_.elements(); i < iend; ++i) {
        s_nonlocal_[i];
    }

    for (int i = 0; i < globals::num_spins; ++i) {
         pos = ::lattice.super_cell_pos(i);
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
}

// --------------------------------------------------------------------------