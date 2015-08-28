#include <cmath>

#include "core/globals.h"
#include "core/consts.h"
#include "core/utils.h"

#include "hamiltonian/dipole_tensor.h"


DipoleHamiltonianTensor::DipoleHamiltonianTensor(const libconfig::Setting &settings)
: HamiltonianStrategy(settings) {
    using std::pow;
    double r_abs;
    jblib::Vec3<double> r_ij, r_hat, s_j;

    jblib::Vec3<int> L_max(0, 0, 0);
    jblib::Vec3<double> super_cell_dim(0.0, 0.0, 0.0);

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice.size(n));
    }

    r_cutoff_ = *std::max_element(super_cell_dim.begin(), super_cell_dim.end());

    if (settings.exists("r_cutoff")) {
        r_cutoff_ = settings["r_cutoff"];
    }


    // printf("  super cell max extent (cartesian):\n    %f %f %f\n", super_cell_dim[0], super_cell_dim[1], super_cell_dim[2]);

    for (int n = 0; n < 3; ++n) {
        if (lattice.is_periodic(n)) {
            L_max[n] = ceil(r_cutoff_/super_cell_dim[n]);
        }
    }

    printf("  image vector max extent (fractional):\n    %d %d %d\n", L_max[0], L_max[1], L_max[2]);

    output.write("  dipole tensor memory estimate (MB):\n    %f\n", std::pow(double(globals::num_spins*3), 2)*8/double(1024*1024) );

    dipole_tensor_.resize(globals::num_spins*3, globals::num_spins*3);

    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < globals::num_spins; ++j) {
            dipole_tensor_(i, j) = 0.0;
        }
    }

    const double prefactor = kVacuumPermeadbility_FourPi*kBohrMagneton/pow(::lattice.parameter(),3);
    jblib::Matrix<double, 3, 3> tensor;

    jblib::Matrix<double, 3, 3> Id( 1, 0, 0, 0, 1, 0, 0, 0, 1 );


    for (int i = 0; i < globals::num_spins; ++i) {

        for (int j = 0; j < globals::num_spins; ++j) {

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

                        // if (i == 0) {
                        // std::cerr << image_vector.x << "\t" << image_vector.y << "\t" << image_vector.z << "\t" <<  r_ij.x << "\t" << r_ij.y << "\t" << r_ij.z << "\t" << lattice.generate_image_position(lattice.position(j), image_vector).x << "\t" << lattice.generate_image_position(lattice.position(j), image_vector).y << "\t" << lattice.generate_image_position(lattice.position(j), image_vector).z << std::endl;
                        // }

                        r_hat = r_ij / r_abs;

                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                tensor[m][n] = 3*r_hat[m]*r_hat[n] - Id[m][n];
                            }
                        }

                        for (int m = 0; m < 3; ++m) {
                            for (int n = 0; n < 3; ++n) {
                                dipole_tensor_(3*i + m, 3*j + n) += tensor[m][n]*prefactor*globals::mus(j)/pow(r_abs,3);
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
   assert(energies.size(0) == globals::num_spins);
   double e_total = 0.0;
   for (int i = 0; i < globals::num_spins; ++i) {
       e_total += calculate_one_spin_energy(i);
   }
    return e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianTensor::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2])*globals::mus(i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianTensor::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianTensor::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    return calculate_one_spin_energy(i, spin_final) - calculate_one_spin_energy(i, spin_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianTensor::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size(0) == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianTensor::calculate_one_spin_field(const int i, double h[3]) {
    int j,m;
    for (m = 0; m < 3; ++m) {
        h[m] = 0.0;
    }
    for (j = 0; j < globals::num_spins3; ++j) {
        for (m = 0; m < 3; ++m) {
            h[m] += dipole_tensor_(3*i + m, j)*globals::s[j];
        }
    }
}


// --------------------------------------------------------------------------

void DipoleHamiltonianTensor::calculate_fields(jblib::Array<double, 2>& energies) {

}

// --------------------------------------------------------------------------