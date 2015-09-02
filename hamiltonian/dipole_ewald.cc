#include "core/globals.h"
#include "core/consts.h"
#include "core/utils.h"

#include "hamiltonian/dipole_ewald.h"

using std::pow;
using std::abs;
using std::min;

DipoleHamiltonianEwald::DipoleHamiltonianEwald(const libconfig::Setting &settings)
: HamiltonianStrategy(settings),
  s_old_(globals::s) {

    double r_abs;
    jblib::Vec3<double> r_ij, eij, s_j;
    jblib::Vec3<int> pos;
    jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

    jblib::Vec3<int> L_max(0, 0, 0);
    jblib::Vec3<double> super_cell_dim(0.0, 0.0, 0.0);

    surf_elec_ = 1.0;

    printf("  Ewald Method\n");

    if (::lattice.num_materials() > 1) {
        jams_error("DipoleHamiltonianEwald only supports single species calculations at the moment");
    }
    local_interaction_matrix_.setMatrixType(SPARSE_MATRIX_TYPE_GENERAL);

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice.size(n));
    }

    delta_error_ = 1e-4;
    if (settings.exists("delta_error")) {
        delta_error_ = settings["delta_error"];
    }

    r_cutoff_ = *std::max_element(super_cell_dim.begin(), super_cell_dim.end());
    if (settings.exists("r_cutoff")) {
        r_cutoff_ = settings["r_cutoff"];
    }
    printf("    r_cutoff: %f\n", r_cutoff_);

    sigma_ = sqrt(-log(delta_error_))/(r_cutoff_);
    if (settings.exists("sigma")) {
        sigma_ = settings["sigma"];
    }
    printf("    sigma: %f\n", sigma_);


    k_cutoff_ = nint(sqrt(-log(delta_error_))*sigma_);
    if (settings.exists("k_cutoff")) {
        k_cutoff_ = settings["k_cutoff"];
    }
    printf("    k_cutoff: %d\n", k_cutoff_);


    // printf("  super cell max extent (cartesian):\n    %f %f %f\n", super_cell_dim[0], super_cell_dim[1], super_cell_dim[2]);

    for (int n = 0; n < 3; ++n) {
        if (lattice.is_periodic(n)) {
            L_max[n] = ceil(r_cutoff_/super_cell_dim[n]);
        }
    }

    printf("  image vector max extent (fractional):\n    %d %d %d\n", L_max[0], L_max[1], L_max[2]);



    local_interaction_matrix_.resize(globals::num_spins*3, globals::num_spins*3);

    const double prefactor = kVacuumPermeadbility_FourPi*kBohrMagneton/pow(::lattice.parameter(),3);

    // --------------------------------------------------------------------------
    // local real space interactions
    // --------------------------------------------------------------------------

    int local_interaction_counter = 0;
    bool is_interacting = false;
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < globals::num_spins; ++j) {
            jblib::Matrix<double, 3, 3> tensor(0, 0, 0, 0, 0, 0, 0, 0, 0);
            is_interacting = false;

            // loop over periodic images of the simulation lattice
            // this means r_cutoff can be larger than the simulation cell
            for (int Lx = -L_max[0]; Lx < L_max[0]+1; ++Lx) {
                for (int Ly = -L_max[1]; Ly < L_max[1]+1; ++Ly) {
                    for (int Lz = -L_max[2]; Lz < L_max[2]+1; ++Lz) {
                        jblib::Vec3<int> image_vector(Lx, Ly, Lz);

                        r_ij  = lattice.generate_image_position(lattice.position(j), image_vector) - lattice.position(i);

                        r_abs = abs(r_ij);

                        // i can interact with i in another image of the simulation cell (just not the 0, 0, 0 image)
                        // so detect based on r_abs rather than i == j
                        if (r_abs > r_cutoff_ || unlikely(r_abs < 1e-5)) continue;

                        is_interacting = true;

                        // if (i == 0) {
                        //     std::cerr << r_ij.x << "\t" << r_ij.y << "\t" << r_ij.z << std::endl;
                        // }

                        eij = r_ij / r_abs;

                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                // tensor[m][n] += (3*eij[m]*eij[n] - Id[m][n])/pow(r_abs, 3);
                                // tensor[m][n] += ((3*eij[m]*eij[n] - Id[m][n])*fG(r_abs, sigma_)/(pow(r_abs, 3)));

                                tensor[m][n] = ((3*eij[m]*eij[n] - Id[m][n])*fG(r_abs, sigma_)/(pow(r_abs, 3)))
                                             - (eij[m]*eij[n]*pG(r_abs, sigma_));
                            }
                        }

                    }
                }
            }
            // std::cerr << i << "\t" << j << "\t" << tensor[0][0]*prefactor*globals::mus(j) << std::endl;
            if (is_interacting) {
                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        local_interaction_matrix_.insertValue(3*i + m, 3*j + n, tensor[m][n]*prefactor*globals::mus(j));
                    }
                }
                // local_interaction_counter++;
            }
        }
    }

    // --------------------------------------------------------------------------

    printf("    total local interactions: %d (%d per spin)\n", local_interaction_counter, local_interaction_counter/globals::num_spins);
    local_interaction_matrix_.convertMAP2CSR();
    printf("    local dipole tensor memory (CSR): %3.2f MB\n", local_interaction_matrix_.calculateMemory());

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

    h_nonlocal_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    s_nonlocal_.resize(kspace_padded_size_[0], kspace_padded_size_[1], kspace_padded_size_[2], 3);
    s_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    h_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3);
    w_recip_.resize(kspace_padded_size_[0], kspace_padded_size_[1], (kspace_padded_size_[2]/2)+1, 3, 3);

    std::fill(w_recip_.data(), w_recip_.data()+w_recip_.elements(), 0.0);

    for (int i = 0, iend = h_nonlocal_.elements(); i < iend; ++i) {
        h_nonlocal_[i] = 0.0;
    }

    for (int i = 0, iend = s_nonlocal_.elements(); i < iend; ++i) {
        s_nonlocal_[i] = 0.0;
    }

    for (int i = 0, iend = s_recip_.elements(); i < iend; ++i) {
        s_recip_[i][0] = 0.0; s_recip_[i][1] = 0.0;
    }

    for (int i = 0, iend = h_recip_.elements(); i < iend; ++i) {
        h_recip_[i][0] = 0.0; h_recip_[i][1] = 0.0;
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

    // divide by product(kspace_padded_size_) instead or normalizing the FFT result later
    const double recip_factor = 4*kPi*kVacuumPermeadbility_FourPi*kBohrMagneton/(lattice.volume()*product(kspace_padded_size_));
    jblib::Vec3<double> kvec;
    double k_abs;
    for (int i = 0; i < kspace_padded_size_[0]; ++i) {
        for (int j = 0; j < kspace_padded_size_[1]; ++j) {
            for (int k = 0; k < (kspace_padded_size_[2]/2) + 1; ++k) {
                pos = jblib::Vec3<int>(i, j, k);

                // convert to +/- r
                for (int n = 0; n < 3; ++n) {
                    if (pos[n] > kspace_padded_size_[n]/2) {
                        pos[n] = periodic_shift(pos[n], kspace_padded_size_[n]/2) - kspace_padded_size_[n]/2;
                    }
                }

                kvec = jblib::Vec3<double>(pos.x, pos.y, pos.z);

                // hack to multiply by inverse lattice vectors
                kvec = lattice.cartesian_to_fractional(kvec);

                k_abs = abs(kvec);

                if (k_abs > k_cutoff_ || unlikely(k_abs < 1e-5)) continue;

                // std::cerr << kvec.x << "\t" << kvec.y << "\t" << kvec.z << std::endl;

                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        w_recip_(i, j, k, m, n) = recip_factor * exp( -0.5 * pow(k_abs / sigma_, 2)) * globals::mus(0)
                                                * kvec[m] * kvec[n] / double(k_abs);
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------

double DipoleHamiltonianEwald::calculate_total_energy() {
    double e_nonlocal = 0.0;
    double e_local = 0.0;
    double e_surface = 0.0;
    double e_self = 0.0;

    double h[3];
    jblib::Vec3<int> pos;

    calculate_nonlocal_ewald_field();
    for (int i = 0; i < globals::num_spins; ++i) {
        pos = ::lattice.super_cell_pos(i);
       e_nonlocal += -(globals::s(i,0)*h_nonlocal_(pos.x, pos.y, pos.z, 0)
                     + globals::s(i,1)*h_nonlocal_(pos.x, pos.y, pos.z, 1)
                     + globals::s(i,2)*h_nonlocal_(pos.x, pos.y, pos.z, 2))*globals::mus(i);
    }

    for (int i = 0; i < globals::num_spins; ++i) {
       calculate_local_ewald_field(i, h);
       e_local += -(globals::s(i,0)*h[0] + globals::s(i,1)*h[1] + globals::s(i,2)*h[2])*globals::mus(i);
    }

    // for (int i = 0; i < globals::num_spins; ++i) {
    //    calculate_self_ewald_field(i, h);
    //    e_self += -0.5*(globals::s(i,0)*h[0] + globals::s(i,1)*h[1] + globals::s(i,2)*h[2])*globals::mus(i);
    // }

    // for (int i = 0; i < globals::num_spins; ++i) {
    //    calculate_surface_ewald_field(i, h);
    //    e_surface += -(globals::s(i,0)*h[0] + globals::s(i,1)*h[1] + globals::s(i,2)*h[2])*globals::mus(i);
    // }

    // std::cerr << e_nonlocal << "\t" << e_local << "\t" << e_surface << "\t" << e_self << std::endl;

    return e_nonlocal + e_local + e_surface + e_self;
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

double DipoleHamiltonianEwald::calculate_one_spin_energy_difference(
    const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final)
{
    jblib::Vec3<int> pos;
    double h[3] = {0.0, 0.0, 0.0};

    calculate_local_ewald_field(i, h);

    calculate_nonlocal_ewald_field();
    for (int m = 0; m < 3; ++m) {
        pos = ::lattice.super_cell_pos(i);
        h[m] += h_nonlocal_(pos.x, pos.y, pos.z, m);
    }

    return (-(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2])
            + (spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]))*globals::mus(i);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianEwald::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.elements() == globals::num_spins);
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

void DipoleHamiltonianEwald::calculate_fields(jblib::Array<double, 2>& fields) {
    double h[3];
    jblib::Vec3<int> pos;

    for (int i = 0; i < globals::num_spins; ++i) {
       calculate_local_ewald_field(i, h);
       for (int j = 0; j < 3; ++j) {
           fields(i, j) = h[j];
        }
    }

    calculate_nonlocal_ewald_field();
    for (int i = 0; i < globals::num_spins; ++i) {
        pos = ::lattice.super_cell_pos(i);
        for (int j = 0; j < 3; ++j) {
            fields(i, j) += h_nonlocal_(pos.x, pos.y, pos.z, j);
        }
    }

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
    int           k;

    for (int m = 0; m < 3; ++m) {
      int begin = ptrb[3*i+m]; int end = ptre[3*i+m];
      for (int j = begin; j < end; ++j) { // j is row
        k = indx[j];                      // k is col
        h[m] = h[m] + globals::s[k]*val[j];
      }
    }
}


void DipoleHamiltonianEwald::calculate_self_ewald_field(const int i, double h[3]) {
    for (int m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }
    for (int n = 0; n < 3; ++n) {
        h[n] += -2*(2.0/(3.0*sqrt(kPi)))*pow(sigma_, 3)*globals::s(i, n)*globals::mus(i);
    }
}

void DipoleHamiltonianEwald::calculate_surface_ewald_field(const int i, double h[3]) {
    for (int m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }

    const double factor = (kBohrMagneton*kBohrMagneton*kTwoPi)/((2.0*surf_elec_ + 1.0)*lattice.volume());
    for (int j = 0; j < globals::num_spins; ++j) {
        for (int n = 0; n < 3; ++n) {
            h[n] += factor*globals::s(j, n)*globals::mus(j);
        }
    }
}

void DipoleHamiltonianEwald::calculate_nonlocal_ewald_field() {
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
         pos = ::lattice.super_cell_pos(i);
         for (int m = 0; m < 3; ++m) {
            s_nonlocal_(pos.x, pos.y, pos.z, m) = globals::s(i, m);
        }
    }

    fftw_execute(spin_fft_forward_transform_);

    // perform convolution as multiplication in fourier space
    for (i = 0, iend = kspace_padded_size_[0]; i < iend; ++i) {
      for (j = 0, jend = kspace_padded_size_[1]; j < jend; ++j) {
        for (k = 0, kend = (kspace_padded_size_[2]/2)+1; k < kend; ++k) {
            for(m = 0; m < 3; ++m) {
                h_recip_(i,j,k,m)[0] = w_recip_(i,j,k,m,0) * s_recip_(i,j,k,0)[0]
                                     + w_recip_(i,j,k,m,1) * s_recip_(i,j,k,1)[0]
                                     + w_recip_(i,j,k,m,2) * s_recip_(i,j,k,2)[0];
                h_recip_(i,j,k,m)[1] = w_recip_(i,j,k,m,0) * s_recip_(i,j,k,0)[1]
                                     + w_recip_(i,j,k,m,1) * s_recip_(i,j,k,1)[1]
                                     + w_recip_(i,j,k,m,2) * s_recip_(i,j,k,2)[1];
                }
            }
        }
      }


    fftw_execute(field_fft_backward_transform_);

    first_run = false;
}

// --------------------------------------------------------------------------