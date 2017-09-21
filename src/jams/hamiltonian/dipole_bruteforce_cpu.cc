#include <omp.h>
#include "jams/core/globals.h"
#include "jams/core/consts.h"
#include "jams/core/utils.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/core/output.h"

#include "jams/hamiltonian/dipole_bruteforce_cpu.h"

DipoleHamiltonianBruteforceCPU::~DipoleHamiltonianBruteforceCPU() {
}

DipoleHamiltonianBruteforceCPU::DipoleHamiltonianBruteforceCPU(const libconfig::Setting &settings, const unsigned int size)
: HamiltonianStrategy(settings, size) {
    Vec3 super_cell_dim = {0.0, 0.0, 0.0};

    for (int n = 0; n < 3; ++n) {
        super_cell_dim[n] = 0.5*double(lattice->size(n));
    }

    settings.lookupValue("r_cutoff", r_cutoff_);
    output->write("  r_cutoff: %e\n", r_cutoff_);

    dipole_prefactor_ = kVacuumPermeadbility*kBohrMagneton /(4*kPi*::lattice->parameter() * ::lattice->parameter() * ::lattice->parameter());
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforceCPU::calculate_total_energy() {
    double e_total = 0.0;


       for (int i = 0; i < globals::num_spins; ++i) {
           e_total += calculate_one_spin_energy(i);
       }

    return e_total;
}

// --------------------------------------------------------------------------


double DipoleHamiltonianBruteforceCPU::calculate_one_spin_energy(const int i, const Vec3 &s_i) {
    double h[3];
    calculate_one_spin_field(i, h);
    return -0.5 * (s_i[0]*h[0] + s_i[1]*h[1] + s_i[2]*h[2]);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforceCPU::calculate_one_spin_energy(const int i) {
    Vec3 s_i = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};
    return calculate_one_spin_energy(i, s_i);
}

// --------------------------------------------------------------------------

double DipoleHamiltonianBruteforceCPU::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    double h[3];
    calculate_one_spin_field(i, h);
    double e_initial = -(spin_initial[0]*h[0] + spin_initial[1]*h[1] + spin_initial[2]*h[2]);
    double e_final = -(spin_final[0]*h[0] + spin_final[1]*h[1] + spin_final[2]*h[2]);
    return 0.5*(e_final - e_initial);
}
// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforceCPU::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}

// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforceCPU::calculate_one_spin_field(const int i, double h[3]) {

    const double w_i = globals::mus(i) * dipole_prefactor_;

    h[0] = 0.0; h[1] = 0.0; h[2] = 0.0;

#pragma omp parallel shared(h)
  {
    Vec3 h_sum = {0.0, 0.0, 0.0};
    #pragma omp for
    for (auto j = 0; j < globals::num_spins; ++j) {
      if (j == i) continue;

      const auto r_ij = lattice->displacement(i, j);
      
      if (abs_sq(r_ij) > pow2(r_cutoff_)) continue;

      const Vec3 s_j = {globals::s(j, 0), globals::s(j, 1), globals::s(j, 2)};

      h_sum += w_i * globals::mus(j) * (3.0 * r_ij * dot(s_j, r_ij) - abs_sq(r_ij) * s_j) / pow5(abs(r_ij));
    }

#pragma omp barrier
#pragma omp critical
    {
      h[0] += h_sum[0];
      h[1] += h_sum[1];
      h[2] += h_sum[2];
    }
  }

}

// --------------------------------------------------------------------------

void DipoleHamiltonianBruteforceCPU::calculate_fields(jblib::Array<double, 2>& fields) {
    for (int i = 0; i < globals::num_spins; ++i) {
        double h[3];

        calculate_one_spin_field(i, h);

        for (int n = 0; n < 3; ++n) {
            fields(i, n) = h[n];
        }
    }
}

void DipoleHamiltonianBruteforceCPU::calculate_fields(jblib::CudaArray<double, 1>& fields) {

}

// --------------------------------------------------------------------------
