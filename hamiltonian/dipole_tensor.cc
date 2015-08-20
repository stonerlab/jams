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
    r_cutoff_ = 1e10;

    output.write("  dipole tensor memory estimate: %fMB", std::pow(double(globals::num_spins*3), 2)*8/double(1024*1024) );

    dipole_tensor_.resize(globals::num_spins*3, globals::num_spins*3);

    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < globals::num_spins; ++j) {
            dipole_tensor_(i, j) = 0.0;
        }
    }

    const double prefactor = kVacuumPermeadbility_FourPi*kBohrMagneton/pow(::lattice.constant(),3);


    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < globals::num_spins; ++j) {
            if (unlikely(j == i)) continue;
            r_ij  = lattice.position(j) - lattice.position(i);
            r_abs = abs(r_ij);

            if (r_abs > r_cutoff_) continue;

            r_hat = r_ij / r_abs;

            jblib::Matrix<double, 3, 3> tensor;

            tensor[0][0] = 3.0*pow(r_hat.x, 2) - 1.0;
            tensor[0][1] = 3.0*r_hat.y*r_hat.x;
            tensor[0][2] = 3.0*r_hat.z*r_hat.x;

            tensor[1][0] = 3.0*r_hat.x*r_hat.y;
            tensor[1][1] = 3.0*pow(r_hat.y, 2) - 1.0;
            tensor[1][2] = 3.0*r_hat.z*r_hat.y;

            tensor[2][0] = 3.0*r_hat.x*r_hat.z;
            tensor[2][1] = 3.0*r_hat.y*r_hat.z;
            tensor[2][2] = 3.0*pow(r_hat.z, 2) - 1.0;

            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    dipole_tensor_(3*i + m, 3*j + n) = tensor[m][n]*prefactor*globals::mus(j)/pow(r_abs,3);
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
    // for (j = 0; j < globals::num_spins3; ++j) {
    //     for (m = 0; m < 3; ++m) {
    //         h[m] += dipole_tensor_(3*i + m, j)*globals::s[j];
    //     }
    // }
    for (j = 0; j < globals::num_spins; ++j) {
        for (m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
                h[m] += dipole_tensor_(3*i + m, j)*globals::s(j, n);
            }
        }
    }

}


// --------------------------------------------------------------------------

void DipoleHamiltonianTensor::calculate_fields(jblib::Array<double, 2>& energies) {

}

// --------------------------------------------------------------------------