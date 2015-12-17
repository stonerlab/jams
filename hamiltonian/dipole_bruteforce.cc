#include "core/globals.h"
#include "core/consts.h"
#include "core/utils.h"

#include "hamiltonian/dipole_bruteforce.h"


DipoleHamiltonianBruteforce::DipoleHamiltonianBruteforce(const libconfig::Setting &settings)
: HamiltonianStrategy(settings) {
    r_cutoff_ = 1e10;
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_total_energy() {
   double e_total = 0.0;
   for (int i = 0; i < globals::num_spins; ++i) {
       e_total += calculate_one_spin_energy(i);
   }
    return e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianBruteforce::calculate_one_spin_energy(const int i, const jblib::Vec3<double> &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2])*globals::mus(i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    return calculate_one_spin_energy(i, spin_final) - calculate_one_spin_energy(i, spin_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforce::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforce::calculate_one_spin_field(const int i, double h[3]) {
    using std::pow;
    int n,j;
    double r_abs, sj_dot_rhat;
    jblib::Vec3<double> r_ij, r_hat;

    const double prefactor = kVacuumPermeadbility*kBohrMagneton/(4*kPi*pow(::lattice.parameter(),3));

    h[0] = 0.0; h[1] = 0.0; h[2] = 0.0;
    for (j = 0; j < globals::num_spins; ++j) {
        if (unlikely(j == i)) continue;

        r_ij  = lattice.atom_position(j) - lattice.atom_position(i);
        r_abs = abs(r_ij);

        if (r_abs > r_cutoff_) continue;

        r_hat = r_ij / r_abs;

        sj_dot_rhat = globals::s(j, 0)*r_hat[0] + globals::s(j, 1)*r_hat[1] + globals::s(j, 2)*r_hat[2];

        for (n = 0; n < 3; ++n) {
            h[n] += ((prefactor*globals::mus(j))/pow(r_abs,3))*(3*r_hat[n]*sj_dot_rhat - globals::s(j, n));
        }
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforce::calculate_fields(jblib::Array<double, 2>& energies) {

}

// --------------------------------------------------------------------------