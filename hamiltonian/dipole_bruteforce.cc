#include "core/globals.h"
#include "core/consts.h"
#include "core/utils.h"

#include "hamiltonian/dipole_bruteforce.h"


DipoleHamiltonianBruteforce::DipoleHamiltonianBruteforce(const libconfig::Setting &settings)
: HamiltonianStrategy(settings) {
    jblib::Vec3<double> super_cell_dim(0.0, 0.0, 0.0);

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice.size(n));
    }

    r_cutoff_ = *std::max_element(super_cell_dim.begin(), super_cell_dim.end());

    settings.lookupValue("r_cutoff", r_cutoff_);
    output.write("  r_cutoff: %e\n", r_cutoff_);

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
    return -(s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_one_spin_energy(const int i) {
    jblib::Vec3<double> s_i(globals::s(i, 0), globals::s(i, 1), globals::s(i, 2));
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforce::calculate_one_spin_energy_difference(const int i, const jblib::Vec3<double> &spin_initial, const jblib::Vec3<double> &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return e_final - e_initial;
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
    int n,j;
    double r_abs, s_j_dot_rhat, w0;
    jblib::Vec3<double> r_ij, r_hat, s_j;

    const double prefactor = globals::mus(i)*kVacuumPermeadbility*kBohrMagneton
                            /(4*kPi*::lattice.parameter() * ::lattice.parameter() * ::lattice.parameter());

    h[0] = 0.0; h[1] = 0.0; h[2] = 0.0;

    for (j = 0; j < globals::num_spins; ++j) {
        if (unlikely(j == i)) continue;

        r_ij = lattice.displacement(i, j);
        r_abs = abs(r_ij);

        if (r_abs > r_cutoff_) continue;

        r_hat = r_ij / r_abs;
        w0 = prefactor * globals::mus(j) / (r_abs * r_abs * r_abs);

        s_j = {globals::s(j, 0), globals::s(j, 1), globals::s(j, 2)};
        s_j_dot_rhat = dot(s_j, r_hat);

        for (n = 0; n < 3; ++n) {
            h[n] += (3 * r_hat[n] * s_j_dot_rhat - s_j[n]) * w0;
        }
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforce::calculate_fields(jblib::Array<double, 2>& energies) {

}

// --------------------------------------------------------------------------