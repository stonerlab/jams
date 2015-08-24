#include "core/globals.h"
#include "core/consts.h"
#include "core/utils.h"

#include "hamiltonian/dipole_Ewald.h"

using std::pow;

DipoleHamiltonianEwald::DipoleHamiltonianEwald(const libconfig::Setting &settings)
: HamiltonianStrategy(settings) {

    ::output.write("  Ewald Method\n");

    if (::lattice.num_materials() > 1) {
        jams_error("DipoleHamiltonianEwald only supports single species calculations at the moment");
    }
    local_interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);

    r_cutoff_ = 5.0;
    ::output.write("    r_cutoff: %f\n", r_cutoff_);


    double r_abs;
    jblib::Vec3<double> r_ij, eij, s_j, pos;
    jblib::Matrix<double, 3, 3> tensor;
    jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

    local_interaction_matrix_.resize(globals::num_spins*3, globals::num_spins*3);

    const double prefactor = kVacuumPermeadbility_FourPi*kBohrMagneton/pow(::lattice.constant(),3);

    const double a = r_cutoff_;

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

            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    tensor[m][n] = ((3*eij[m]*eij[n] - Id[m][n])*fG(r_abs, a)/(pow(r_abs, 3)))
                                 - (2*eij[m]*eij[n]*gaussian(r_abs, a)/(a*a));
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


    // --------------------------------------------------------------------------
    // nonlocal reciprocal space interactions
    // --------------------------------------------------------------------------
    jblib::Array<double, 5> wij(::lattice.num_unit_cells(0),
                                ::lattice.num_unit_cells(1),
                                ::lattice.num_unit_cells(2),
                                3,
                                3);

    std::fill(wij.data(), wij.data()+::lattice.num_unit_cells(0)*::lattice.num_unit_cells(1)*::lattice.num_unit_cells(2)*3*3, 0.0);

    jblib::Vec3<int> kdim(::lattice.num_unit_cells(0), ::lattice.num_unit_cells(1), ::lattice.num_unit_cells(2));

    for (int i = 0; i < kdim.x; ++i) {
        for (int j = 0; j < kdim.y; ++j) {
            for (int k = 0; k < kdim.z; ++k) {
                if (unlikely(i == 0 && j == 0 && k == 0)) continue;

                // need to account for -/+ minimum image convention
                pos = jblib::Vec3<double>(i, j, k);

                if ( i > kdim.x/2 + 1) {
                    pos.x = periodic_shift(pos.x, kdim.x/2) - kdim.x/2;
                }

                if ( j > kdim.y/2 + 1) {
                    pos.y = periodic_shift(pos.y, kdim.y/2) - kdim.y/2;
                }

                if ( k > kdim.z/2 + 1) {
                    pos.z = periodic_shift(pos.z, kdim.z/2) - kdim.z/2;
                }

                r_ij  = ::lattice.fractional_to_cartesian_position(pos);
                r_abs = abs(r_ij);

                if (r_abs > r_cutoff_) continue;

                eij = r_ij / r_abs;

                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        // divide by num_spins instead or normalizing the FFT result later
                        wij(i, j, k, m, n) += globals::mus(0)*prefactor*(2*eij[m]*eij[n]*gaussian(r_abs, a)/(globals::num_spins*a*a));
                    }
                }
            }
        }
    }


    h_nonlocal_.resize(wij.size(0), wij.size(1), wij.size(2), 3);
    s_recip_.resize(wij.size(0), wij.size(1), (wij.size(2)/2)+1, 3);
    h_recip_.resize(wij.size(0), wij.size(1), (wij.size(2)/2)+1, 3);
    w_recip_.resize(wij.size(0), wij.size(1), (wij.size(2)/2)+1, 3, 3);

    const int kspace_dimensions[3] = {wij.size(0), wij.size(1), wij.size(2)};

    ::output.write("    kspace dimensions: %d %d %d\n", wij.size(0), wij.size(1), wij.size(2));

    ::output.write("    planning FFTs\n");

    spin_fft_forward_transform_   = fftw_plan_many_dft_r2c(3, kspace_dimensions, 3, globals::s.data(),  NULL, 3, 1, s_recip_.data(), NULL, 3, 1, FFTW_PATIENT|FFTW_PRESERVE_INPUT);
    field_fft_backward_transform_ = fftw_plan_many_dft_c2r(3, kspace_dimensions, 3, h_recip_.data(), NULL, 3, 1, h_nonlocal_.data(),  NULL, 3, 1, FFTW_PATIENT);
    interaction_fft_transform_    = fftw_plan_many_dft_r2c(3, kspace_dimensions, 9, wij.data(),  NULL, 9, 1, w_recip_.data(), NULL, 9, 1, FFTW_MEASURE|FFTW_PRESERVE_INPUT);

    ::output.write("    transform interaction matrix\n");

    fftw_execute(interaction_fft_transform_);
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
        for (int m = 0; m < 3; ++m) {

           e_total += -(globals::s(i,0)*h_nonlocal_(pos.x, pos.y, pos.z, 0)
                    + globals::s(i,1)*h_nonlocal_(pos.x, pos.y, pos.z, 1)
                    + globals::s(i,2)*h_nonlocal_(pos.x, pos.y, pos.z, 2))*globals::mus(i);
        }
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
    return calculate_one_spin_energy(i, spin_final) - calculate_one_spin_energy(i, spin_initial);
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
    int i, iend, j, jend, k, kend, m;

    fftw_execute(spin_fft_forward_transform_);

    // perform convolution as multiplication in fourier space
    for (i = 0, iend = w_recip_.size(0); i < iend; ++i) {
      for (j = 0, jend = w_recip_.size(1); j < jend; ++j) {
        for (k = 0, kend = w_recip_.size(2); k < kend; ++k) {
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

    fftw_execute(field_fft_backward_transform_);

    // for (i = 0; i < globals::num_spins3; ++i) {
    //   h_nonlocal_[i] /= static_cast<double>(globals::num_spins);
    // }
}

// --------------------------------------------------------------------------