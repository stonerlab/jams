#include <cassert>
#include <cmath>
#include <cstdio>
#include <algorithm>

#include "jams/core/lattice.h"
#include "jams/interface/blas.h"
#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"

#include "dipole_tensor.h"


DipoleHamiltonianTensor::DipoleHamiltonianTensor(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size) {
    using std::pow;
    double r_abs;
    Vec3 r_ij, r_hat, s_j;

    Vec3i L_max = {0, 0, 0};
    Vec3 super_cell_dim = {0.0, 0.0, 0.0};

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice->size(n));
    }

    r_cutoff_ = settings["r_cutoff"];

    // printf("  super cell max extent (cartesian):\n    %f %f %f\n", super_cell_dim[0], super_cell_dim[1], super_cell_dim[2]);

    for (int n = 0; n < 3; ++n) {
        if (lattice->is_periodic(n)) {
            L_max[n] = ceil(r_cutoff_/super_cell_dim[n]);
        }
    }

    std::cout << "  image vector max extent (fractional) " << L_max[0] << " " << L_max[1] << " " << L_max[2] << "\n";

    std::cout << "  dipole tensor memory estimate " << std::pow(double(globals::num_spins*3), 2)*8/double(1024*1024) << "(MB)\n";

    dipole_tensor_ = jblib::Array<double,2>(globals::num_spins3, globals::num_spins3);
    dipole_tensor_.zero();

    const double prefactor = kVacuumPermeadbility*kBohrMagneton/(4*kPi*pow(::lattice->parameter(),3));


    for (int i = 0; i < globals::num_spins; ++i) {

        for (int j = 0; j < globals::num_spins; ++j) {

            // loop over periodic images of the simulation lattice
            // this means r_cutoff can be larger than the simulation cell
            for (int Lx = -L_max[0]; Lx < L_max[0]+1; ++Lx) {
                for (int Ly = -L_max[1]; Ly < L_max[1]+1; ++Ly) {
                    for (int Lz = -L_max[2]; Lz < L_max[2]+1; ++Lz) {
                        Vec3i image_vector = {Lx, Ly, Lz};

                        r_ij  = lattice->generate_image_position(lattice->atom_position(j), image_vector) - lattice->atom_position(i);

                        r_abs = abs(r_ij);

                        // i can interact with i in another image of the simulation cell (just not the 0, 0, 0 image)
                        // so detect based on r_abs rather than i == j
                      if (definately_greater_than(r_abs, r_cutoff_, jams::defaults::lattice_tolerance) || unlikely(approximately_zero(r_abs, jams::defaults::lattice_tolerance))) continue;

                        r_hat = r_ij / r_abs;

                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                dipole_tensor_(3*i + m, 3*j + n) += (3*r_hat[m]*r_hat[n] - kIdentityMat3[m][n])*prefactor*globals::mus(i)*globals::mus(j)/pow(r_abs,3);
                            }
                        }
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------

double DipoleHamiltonianTensor::calculate_total_energy() {
   double e_total = 0.0;
   for (int i = 0; i < globals::num_spins; ++i) {
       e_total += calculate_one_spin_energy(i);
   }
    return 0.5*e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianTensor::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianTensor::calculate_one_spin_energy(const int i) {
    return calculate_one_spin_energy(i, {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)});
}

// --------------------------------------------------------------------------

double DipoleHamiltonianTensor::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return 0.5*(e_final - e_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianTensor::calculate_energies(jams::MultiArray<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies(i) = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianTensor::calculate_one_spin_field(const int i, double h[3]) {
    // int j,m;
    // for (m = 0; m < 3; ++m) {
    //     h[m] = 0.0;
    // }
    // for (j = 0; j < globals::num_spins3; ++j) {
    //     for (m = 0; m < 3; ++m) {
    //         h[m] += dipole_tensor_(3*i + m, j)*globals::s[j];
    //     }
    // }

    cblas_dgemv(
        CblasRowMajor,          // storage order
        CblasNoTrans,           // transpose?
        3,                      // m rows
        globals::num_spins3,    // n cols
        1.0,                    // alpha
        dipole_tensor_.data()+(3*i*globals::num_spins3),   // A matrix
        globals::num_spins3,    // first dimension of A
        globals::s.data(),       // x vector
        1,                      // increment of x
        0.0,                    // beta
        &h[0],           // y vector
        1                       // increment of y
        );
}


// --------------------------------------------------------------------------

void DipoleHamiltonianTensor::calculate_fields(jams::MultiArray<double, 2>& fields) {
    // int i, j, m, n;
    // for (i = 0; i < globals::num_spins; ++i) {
    //     fields(i, 0) = 0.0; fields(i, 1) = 0.0; fields(i, 2) = 0.0;
    //     for (j = 0; j < globals::num_spins; ++j) {
    //         for (m = 0; m < 3; ++m) {
    //             for (n = 0; n < 3; ++n) {
    //                 fields(i, m) += dipole_tensor_(3*i + m, 3*j + n)*globals::s(j, n);
    //             }
    //         }
    //     }
    // }

    // // y := alpha*A*x + beta*y
    cblas_dgemv(
        CblasRowMajor,          // storage order
        CblasNoTrans,           // transpose?
        globals::num_spins3,    // m rows
        globals::num_spins3,    // n cols
        1.0,                    // alpha
        dipole_tensor_.data(),   // A matrix
        globals::num_spins3,    // first dimension of A
        globals::s.data(),       // x vector
        1,                      // increment of x
        0.0,                    // beta
        fields.data(),           // y vector
        1                       // increment of y
        );
}

// --------------------------------------------------------------------------
