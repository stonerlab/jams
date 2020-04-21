#ifdef HAS_OMP
#include <omp.h>
#endif

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"

#include "dipole_bruteforce.h"

DipoleHamiltonianCpuBruteforce::~DipoleHamiltonianCpuBruteforce() {
}

DipoleHamiltonianCpuBruteforce::DipoleHamiltonianCpuBruteforce(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size) {

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

    int num_neighbours = 0;
    for (auto i = 0; i < globals::num_spins; ++i) {
      num_neighbours += lattice->num_neighbours(i, r_cutoff_);
    }

    std::cout << "  num_neighbours " << num_neighbours << "\n";

}

// --------------------------------------------------------------------------

double DipoleHamiltonianCpuBruteforce::calculate_total_energy() {
    double e_total = 0.0;

       for (auto i = 0; i < globals::num_spins; ++i) {
           e_total += calculate_one_spin_energy(i);
       }

    return e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianCpuBruteforce::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -0.5 * (s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCpuBruteforce::calculate_one_spin_energy(const int i) {
    Vec3 s_i = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianCpuBruteforce::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return 0.5*(e_final - e_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianCpuBruteforce::calculate_energies() {
    for (auto i = 0; i < globals::num_spins; ++i) {
        energy_(i) = calculate_one_spin_energy(i);
    }
}


__attribute__((hot))
void DipoleHamiltonianCpuBruteforce::calculate_one_spin_field(const int i, double h[3])
{
  using namespace globals;

  const auto w0 = kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * pow3(lattice->parameter()));

  double hx = 0, hy = 0, hz = 0;

  const Vec3 r_i = lattice->atom_position(i);

  const auto neighbours = lattice->atom_neighbours(i, r_cutoff_);
  #if HAS_OMP
  #pragma omp parallel for reduction(+:hx, hy, hz)
  #endif
  for (auto n = 0; n < neighbours.size(); ++n) {
    const int j = neighbours[n].second;
    if (j == i) continue;

    const Vec3 r_ij =  neighbours[n].first - r_i;

    const auto r_abs = norm(r_ij);

    const auto sj_dot_r = s(j, 0) * r_ij[0] + s(j, 1) * r_ij[1] + s(j, 2) * r_ij[2];

    hx += w0 * mus(i) * mus(j) * (3.0 * r_ij[0] * sj_dot_r - pow2(r_abs) * s(j, 0)) / pow5(r_abs);
    hy += w0 * mus(i) * mus(j) * (3.0 * r_ij[1] * sj_dot_r - pow2(r_abs) * s(j, 1)) / pow5(r_abs);
    hz += w0 * mus(i) * mus(j) * (3.0 * r_ij[2] * sj_dot_r - pow2(r_abs) * s(j, 2)) / pow5(r_abs);
  }

  h[0] = hx; h[1] = hy; h[2] = hz;
}

// --------------------------------------------------------------------------

void DipoleHamiltonianCpuBruteforce::calculate_fields() {
    for (auto i = 0; i < globals::num_spins; ++i) {
        double h[3];

        calculate_one_spin_field(i, h);

        for (auto n : {0,1,2}) {
            field_(i, n) = h[n];
        }
    }
}