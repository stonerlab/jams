#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"

#include "jams/hamiltonian/zeeman.h"
#include <jams/helpers/exception.h>

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
        if (settings["dc_local_field"].getLength() != globals::lattice->num_materials()) {
          throw jams::ConfigException(settings["dc_local_field"], "field must be specified for every material");
        }


        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                dc_local_field_(i, j) = settings["dc_local_field"][globals::lattice->lattice_site_material_id(i)][j];
                dc_local_field_(i, j) *= globals::mus(i);
            }
        }
    }

    if(settings.exists("ac_local")) {
        if (settings["ac_local"].getLength() != globals::lattice->num_materials()) {
          throw jams::ConfigException(settings["ac_local"], "field must be specified for every material");
        }
    }

    has_ac_local_field_ = false;
    if(settings.exists("ac_local_field") || settings.exists("ac_local_frequency")) {
        if(!(settings.exists("ac_local_field") && settings.exists("ac_local_frequency"))) {
          throw jams::ConfigException(settings["ac_local_field"], "must have a field and a frequency");
        }
        if (settings["ac_local_frequency"].getLength() != globals::lattice->num_materials()) {
          throw jams::ConfigException(settings["ac_local_frequency"], "must be specified for every material");
        }
        if (settings["ac_local_field"].getLength() != globals::lattice->num_materials()) {
          throw jams::ConfigException(settings["ac_local_field"], "must be specified for every material");
        }

        has_ac_local_field_ = true;

        for (int i = 0; i < globals::num_spins; ++i) {
            for (int j = 0; j < 3; ++j) {
                ac_local_field_(i, j) = settings["ac_local_field"][globals::lattice->lattice_site_material_id(i)][j];
                ac_local_field_(i, j) *= globals::mus(i);
            }
        }

        for (int i = 0; i < globals::num_spins; ++i) {
            ac_local_frequency_(i) = settings["ac_local_frequency"][globals::lattice->lattice_site_material_id(i)];
            ac_local_frequency_(i) = kTwoPi*ac_local_frequency_(i);
        }
    }
}

double ZeemanHamiltonian::calculate_total_energy(double time) {
    double e_total = 0.0;
    for (int i = 0; i < globals::num_spins; ++i) {
        e_total += calculate_energy(i, time);
    }
     return e_total;
}

double ZeemanHamiltonian::calculate_energy(const int i, double time) {
    const Vec3 s_i = {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
    const auto field = calculate_field(i, time);

    return -dot(s_i, field);
}

double ZeemanHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) {
  const auto field = calculate_field(i, time);
  const auto e_initial = -dot(spin_initial, field);
  const auto e_final = -dot(spin_initial, field);

  return (e_final - e_initial);
}

void ZeemanHamiltonian::calculate_energies(double time) {
    for (int i = 0; i < globals::num_spins; ++i) {
        energy_(i) = calculate_energy(i, time);
    }
}

Vec3 ZeemanHamiltonian::calculate_field(const int i, double time) {
    using std::pow;

    Vec3 field = {0.0, 0.0, 0.0};

    for (int j = 0; j < 3; ++j) {
      field[j] = dc_local_field_(i, j);
    }

    if (has_ac_local_field_) {
        for (int j = 0; j < 3; ++j) {
          field[j] += ac_local_field_(i, j) * cos(ac_local_frequency_(i) * time);
        }
    }

    return field;
}

void ZeemanHamiltonian::calculate_fields(double time) {
    for (int i = 0; i < globals::num_spins; ++i) {
        for (int j = 0; j < 3; ++j) {
            field_(i, j) = dc_local_field_(i, j);
        }
        if (has_ac_local_field_) {
            for (int j = 0; j < 3; ++j) {
                field_(i, j) += ac_local_field_(i, j) * cos(ac_local_frequency_(i) * time);
            }
        }
    }
}
