#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"

#include "jams/hamiltonian/zeeman.h"

ZeemanHamiltonian::ZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size)
{
    dc_local_field_.resize(globals::num_spins, 3);
    dc_local_field_.zero();


    ac_local_field_.resize(globals::num_spins, 3);
    ac_local_frequency_.resize(globals::num_spins);

    ac_local_field_.zero();
    ac_local_frequency_.zero();


    if(settings.exists("dc_local_field")) {
        if (settings["dc_local_field"].getLength() != lattice->num_materials()) {
          jams_die("ZeemanHamiltonian: dc_local_field must be specified for every material");
        }


        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                dc_local_field_(i, j) = settings["dc_local_field"][lattice->atom_material_id(i)][j];
                dc_local_field_(i, j) *= globals::mus(i);
            }
        }
    }

    if(settings.exists("ac_local")) {
        if (settings["ac_local"].getLength() != lattice->num_materials()) {
          jams_die("ZeemanHamiltonian: ac_local must be specified for every material");
        }
    }

    has_ac_local_field_ = false;
    if(settings.exists("ac_local_field") || settings.exists("ac_local_frequency")) {
        if(!(settings.exists("ac_local_field") && settings.exists("ac_local_frequency"))) {
          jams_die("ZeemanHamiltonian: ac_local must have a field and a frequency");
        }
        if (settings["ac_local_frequency"].getLength() != lattice->num_materials()) {
          jams_die("ZeemanHamiltonian: ac_local_frequency must be specified for every material");
        }
        if (settings["ac_local_field"].getLength() != lattice->num_materials()) {
          jams_die("ZeemanHamiltonian: ac_local_field must be specified for every material");
        }

        has_ac_local_field_ = true;

        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                ac_local_field_(i, j) = settings["ac_local_field"][lattice->atom_material_id(i)][j];
                ac_local_field_(i, j) *= globals::mus(i);
            }
        }

        for (int i = 0; i < globals::num_spins; ++i) {
            ac_local_frequency_(i) = settings["ac_local_frequency"][lattice->atom_material_id(i)];
            ac_local_frequency_(i) = kTwoPi*ac_local_frequency_(i);
        }
    }
}

double ZeemanHamiltonian::calculate_total_energy() {
    double e_total = 0.0;
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += calculate_one_spin_energy(i);
    }
     return e_total;
}

double ZeemanHamiltonian::calculate_one_spin_energy(const int i) {
    using namespace globals;
    double one_spin_field[3];

    calculate_one_spin_field(i, one_spin_field);

    return -(s(i, 0)*one_spin_field[0] + s(i, 1)*one_spin_field[1] + s(i, 2)*one_spin_field[2]);
}

double ZeemanHamiltonian::calculate_one_spin_energy_difference(const int i, const Vec3 &spin_initial, const Vec3 &spin_final) {
    using std::pow;

    double e_initial = 0.0;
    double e_final = 0.0;

    double h_local[3];

    calculate_one_spin_field(i, h_local);

    for (int n = 0; n < 3; ++n) {
        e_initial += -spin_initial[n]*h_local[n];
    }

    for (int n = 0; n < 3; ++n) {
        e_final += -spin_final[n]*h_local[n];
    }

    return (e_final - e_initial);
}

void ZeemanHamiltonian::calculate_energies() {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_[i] = calculate_one_spin_energy(i);
    }
}

void ZeemanHamiltonian::calculate_one_spin_field(const int i, double local_field[3]) {
    using namespace globals;
    using std::pow;

    for (int j = 0; j < 3; ++j) {
        local_field[j] = dc_local_field_(i, j);
    }

    if (has_ac_local_field_) {
        for (int j = 0; j < 3; ++j) {
            local_field[j] += ac_local_field_(i, j) * cos(ac_local_frequency_(i) * solver->time());
        }
    }
}

void ZeemanHamiltonian::calculate_fields() {
  for (int i = 0; i < globals::num_spins; ++i) {
      for (int j = 0; j < 3; ++j) {
          field_(i, j) = dc_local_field_(i, j) + ac_local_field_(i, j) * cos(ac_local_frequency_(i) * solver->time());
      }
    }
}
