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
: HamiltonianStrategy(settings, size) {

    settings.lookupValue("r_cutoff", r_cutoff_);
    std::cout << "  r_cutoff " << r_cutoff_ << "\n";

  supercell_matrix_ = lattice->get_supercell().matrix();

  frac_positions_.resize(globals::num_spins);

    for (auto i = 0; i < globals::num_spins; ++i) {
      frac_positions_[i] = lattice->get_supercell().inverse_matrix()*lattice->atom_position(i);
    }

  }

// --------------------------------------------------------------------------

double DipoleHamiltonianCpuBruteforce::calculate_total_energy() {
    double e_total = 0.0;

       for (int i = 0; i < globals::num_spins; ++i) {
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

void DipoleHamiltonianCpuBruteforce::calculate_energies(jblib::Array<double, 1>& energies) {
    assert(energies.size() == globals::num_spins);
    for (int i = 0; i < globals::num_spins; ++i) {
        energies[i] = calculate_one_spin_energy(i);
    }
}


__attribute__((hot))
void DipoleHamiltonianCpuBruteforce::calculate_one_spin_field(const int i, double h[3])
{
  using namespace globals;

  const auto r_cut_squared = pow2(r_cutoff_);
  const auto w0 = kVacuumPermeadbility * kBohrMagneton / (4.0 * kPi * pow3(lattice->parameter()));

  const bool is_bulk = lattice->is_periodic(0) && lattice->is_periodic(1) && lattice->is_periodic(2);
  const bool is_open = !lattice->is_periodic(0) && !lattice->is_periodic(1) && !lattice->is_periodic(2);

  double hx = 0, hy = 0, hz = 0;
#pragma omp parallel for reduction(+:hx, hy, hz)
  for (auto j = 0; j < globals::num_spins; ++j) {
    if (j == i) continue;

    Vec3 r_ij = frac_positions_[j] - frac_positions_[i];

    if (likely(is_bulk)) {
      r_ij = supercell_matrix_ * (r_ij - trunc(2 * r_ij));
    } else if (!is_open) {
      for (auto n = 0; n < 3; ++n) {
        if (lattice->is_periodic(n)) {
          r_ij[n] = r_ij[n] - trunc(2.0 * r_ij[n]);
        }
      }
      r_ij = supercell_matrix_ * r_ij;
    }

    auto r_abs_sq = abs_sq(r_ij);

    if (r_abs_sq > r_cut_squared) continue;

    auto sj_dot_r = s(j, 0) * r_ij[0] + s(j, 1) * r_ij[1] + s(j, 2) * r_ij[2];

    hx += w0 * mus(i) * mus(j) * (3 * r_ij[0] * sj_dot_r - r_abs_sq * s(j, 0)) / pow(r_abs_sq, 2.5);
    hy += w0 * mus(i) * mus(j) * (3 * r_ij[1] * sj_dot_r - r_abs_sq * s(j, 1)) / pow(r_abs_sq, 2.5);
    hz += w0 * mus(i) * mus(j) * (3 * r_ij[2] * sj_dot_r - r_abs_sq * s(j, 2)) / pow(r_abs_sq, 2.5);
  }

  h[0] = hx; h[1] = hy; h[2] = hz;
}

// --------------------------------------------------------------------------

void DipoleHamiltonianCpuBruteforce::calculate_fields(jblib::Array<double, 2>& fields) {
    for (int i = 0; i < globals::num_spins; ++i) {
        double h[3];

        calculate_one_spin_field(i, h);

        for (int n = 0; n < 3; ++n) {
            fields(i, n) = h[n];
        }
    }
}