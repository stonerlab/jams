#include "jams/core/globals.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/error.h"

#include "jams/hamiltonian/zeeman.h"
#include <jams/helpers/exception.h>
#include <jams/interface/config.h>

ZeemanHamiltonian::ZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size)
: Hamiltonian(settings, size)
{
    dc_local_field_.resize(globals::num_spins, 3);
    dc_local_field_.zero();


    ac_local_field_.resize(globals::num_spins, 3);
    ac_local_frequency_.resize(globals::num_spins);

    ac_local_field_.zero();
    ac_local_frequency_.zero();


    const auto num_materials = globals::lattice->num_materials();

    if(settings.exists("dc_local_field")) {
        const auto dc_fields = jams::read_vec_sequence_setting<jams::Real, 3>(
            settings["dc_local_field"], "dc_local_field", num_materials);
        for (int i = 0; i < globals::num_spins; ++i) {
            const auto& field = dc_fields[globals::lattice->lattice_site_material_id(i)];
            for (int j = 0; j < 3; ++j) {
                dc_local_field_(i, j) = field[j] * globals::mus(i);
            }
        }
    }

    if(settings.exists("ac_local")) {
        jams::require_setting_length(settings["ac_local"], "ac_local", num_materials);
    }

    has_ac_local_field_ = false;
    jams::require_settings_together(settings, {"ac_local_field", "ac_local_frequency"});
    if(settings.exists("ac_local_field")) {
        has_ac_local_field_ = true;
        const auto ac_fields = jams::read_vec_sequence_setting<jams::Real, 3>(
            settings["ac_local_field"], "ac_local_field", num_materials);
        const auto ac_frequencies = jams::read_numeric_sequence_setting<jams::Real>(
            settings["ac_local_frequency"], "ac_local_frequency", num_materials);

        for (int i = 0; i < globals::num_spins; ++i) {
            const auto material_id = globals::lattice->lattice_site_material_id(i);
            const auto& field = ac_fields[material_id];
            for (int j = 0; j < 3; ++j) {
                ac_local_field_(i, j) = field[j] * globals::mus(i);
            }
            ac_local_frequency_(i) = kTwoPi * ac_frequencies[material_id];
        }
    }
}


jams::Real ZeemanHamiltonian::calculate_energy(const int i, jams::Real time) {
    return calculate_energy_for_spin(i, {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)}, time);
}

jams::Real ZeemanHamiltonian::calculate_energy_for_spin(const int i, const jams::Vec<double, 3>& spin, jams::Real time) {
    const auto field = calculate_field(i, time);

    return -jams::dot(jams::array_cast<jams::Real>(spin), field);
}


jams::Vec<jams::Real, 3> ZeemanHamiltonian::calculate_field(const int i, jams::Real time) {
    using std::pow;

    jams::Vec<jams::Real, 3> field = {0.0, 0.0, 0.0};

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
