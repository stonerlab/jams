//
// Created by Sean Stansill [ll14s26s] on 29/03/2023.
//

#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/hamiltonian/uniaxial_generalised.h"

using libconfig::Setting;
using std::string;
using std::runtime_error;
using std::vector;

// Store anisotropy settings in an intermediate structure defined per material
// before added to per spin containers
struct AnisotropySettings {
    // Atom position (material number)
    int      motif_position = -1;

    // Material name if specified instead of material number
    string   material = "";

    // Vector u in the documentation
    Vec3     axis1 = {1.0, 0.0, 0.0};

    // Vector v in the documentation
    Vec3     axis2 = {0.0, 1.0, 0.0};

    // Vector w in the documentation
    Vec3     axis3 = {0.0, 0.0, 1.0};

    // The magnitude of the anisotropy constant
    double   energy = 0.0;
};

// Helper functions to be used in the constructor

// Turns an anisotropy in into the struct above
AnisotropySettings read_uniaxial_setting(Setting &setting) {
    // Anisotropies must be a list
    if (!setting.isList()) {
        throw runtime_error("Incorrectly formatted anisotropy setting");
    }
    AnisotropySettings result;

    // If materials are listed by number
    if (setting[0].isNumber()) {
        result.motif_position = int(setting[0]);
        result.motif_position--;
        if (result.motif_position < 0 || result.motif_position >= globals::lattice->num_motif_atoms()) {
            throw runtime_error("uniaxial generalised anisotropy motif position is invalid");
        }
    }
    // Else the materials must be listed by name
    else {
        result.material = string(setting[0].c_str());
        if (!globals::lattice->material_exists(result.material)) {
            throw runtime_error("uniaxial generalised anisotropy material is invalid");
        }
    }

    // Populate other required information
    result.axis1 = normalize(Vec3{setting[1][0], setting[1][1], setting[1][2]});
    result.axis2 = normalize(Vec3{setting[2][0], setting[2][1], setting[2][2]});
    result.axis3 = normalize(Vec3{setting[3][0], setting[3][1], setting[3][2]});
    result.energy = double(setting[4]);
    return result;
}

// Loop over the anisotropies. Must be same length as the number of materials
vector<AnisotropySettings> read_all_uniaxial_settings(const Setting &settings) {
    vector<AnisotropySettings> anisotropies;
    for (auto i = 0; i < settings["anisotropies"].getLength(); ++i) {
        anisotropies.push_back(read_uniaxial_setting(settings["anisotropies"][i]));
    }
    return anisotropies;
}


// Constructor
UniaxialGeneralisedHamiltonian::UniaxialGeneralisedHamiltonian(const Setting &settings, const unsigned int num_spins)
        : Hamiltonian(settings, num_spins) {

    // Check settings contains the identifier setting
    // This tells us which anisotropy term the user is specifying
    Vec3i identifier = jams::config_required<Vec3i>(settings, "identifier");

    // Use identifier to set required anisotropy coefficient to 1.0
    // Second order anisotorpy
    if (identifier[0] == 2) {
        if (identifier[1] == 0 && identifier[2] == 1) {
            jams_die("Anisotropy identifier not recognised");
        } else {
            a1_ = 1.0; a2_ = 0.0; a3_ = 0.0; a4_ = 0.0; a5_ = 0.0; a6_ = 0.0;
        }
    }
    // Fourth order anisotropies
    else if (identifier[0] == 4) {
        if (identifier[1] == 0 && identifier[2] == 0) {
            a1_ = 0.0; a2_ = 1.0; a3_ = 0.0; a4_ = 0.0; a5_ = 0.0; a6_ = 0.0;
        } else if (identifier[1] == 3 && identifier[2] == 0) {
            a1_ = 0.0; a2_ = 0.0; a3_ = 1.0; a4_ = 0.0; a5_ = 0.0; a6_ = 0.0;
        } else if (identifier[1] == 4) {
            if (identifier[2] == 0) {
                a1_ = 0.0; a2_ = 0.0; a3_ = 0.0; a4_ = 1.0; a5_ = 0.0; a6_ = 0.0;
            } else if (identifier[2] == 1) {
                a1_ = 0.0; a2_ = 0.0; a3_ = 0.0; a4_ = 0.0; a5_ = 1.0; a6_ = 0.0;
            } else if (identifier[2] == 2) {
                a1_ = 0.0; a2_ = 0.0; a3_ = 0.0; a4_ = 0.0; a5_ = 0.0; a6_ = 1.0;
            } else {
                jams_die("Anisotropy identifier not recognised");
            }
        }
    }

    // Read in anisotropy constants and axes
    auto anisotropies = read_all_uniaxial_settings(settings);

    // Resize arrays
    zero(magnitude_.resize(num_spins));
    zero(axis1_.resize(num_spins));
    zero(axis2_.resize(num_spins));
    zero(axis3_.resize(num_spins));

    // Loop over the different materials / anisotropies
    for (const auto& ani : anisotropies) {
        // Loop over all atomss to populate per spin containers
        for (auto i = 0; i < globals::num_spins; ++i) {
            // If anisotropies were inputted by material number
            if (globals::lattice->atom_motif_position(i) == ani.motif_position) {
                magnitude_(i) = ani.energy * input_energy_unit_conversion_;
                axis1_(i) = ani.axis1;
                axis2_(i) = ani.axis2;
                axis3_(i) = ani.axis3;
            }
            // If anisotropies were inputted by material name
            if (globals::lattice->material_exists(ani.material)) {
                if (globals::lattice->atom_material_id(i) == globals::lattice->material_id(ani.material)) {
                    magnitude_(i) = ani.energy * input_energy_unit_conversion_;
                    axis1_(i) = ani.axis1;
                    axis2_(i) = ani.axis2;
                    axis3_(i) = ani.axis3;
                }
            }
        }
    }
}


double UniaxialGeneralisedHamiltonian::calculate_total_energy(double time) {
    double e_total = 0.0;
    for (int i = 0; i < energy_.size(); ++i) {
        e_total += calculate_energy(i, time);
    }
    return e_total;
}

double UniaxialGeneralisedHamiltonian::calculate_energy(const int i, double time) {
    double energy = 0.0;
    Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};

    // Calculate the dot product of the spin vector with the anisotropy axes
    auto Su = dot(axis1_(i), spin);
    auto Sv = dot(axis2_(i), spin);
    auto Sw = dot(axis3_(i), spin);

    // Some terms appear multiple times. To reduce number of multiplications
    // by storing them here
    // FUTURE OPTIMIZATION: Use one line ifs to determine whether to do the
    // calculations. Needs profiling to determine if necessary.
    auto Su2 = pow(Su, 2);
    auto Sv2 = pow(Sv, 2);
    auto Sw2 = pow(Sw, 2);

    // Terms are
    // Sw**2
    // Sw**4
    // Sw*Sv*( 3*Su**2 - Sv**2 )
    // Su**4 + Sv**4
    // Su**2 * Sv**2
    // Su**4 - 6*Su**2*Sv**2 + Sv**4
    energy += -magnitude_(i) * ( a1_*Sw2 + a2_*pow2(Sw2) + a3_*Sw*Sv*(3*Su2-Sv2) +
                a4_*(pow2(Su2)+pow2(Sv2)) + a5_*Su2*Sv2 + a6_*(pow2(Su2)-6*Su2*Sv2+pow2(Sv2)) );

    return energy;

}

double UniaxialGeneralisedHamiltonian::calculate_energy_difference(int i, const Vec3 &spin_initial,
        const Vec3 &spin_final, double time) {

    // Initialize containers
    double e_initial = 0.0;
    double e_final = 0.0;

    /////////////////////////////////////////////////////
    ///////////// INITIAL ///////////////////////////////
    auto Su_initial = dot(axis1_(i), spin_initial);
    auto Sv_initial = dot(axis2_(i), spin_initial);
    auto Sw_initial = dot(axis3_(i), spin_initial);

    auto Su2_initial = pow(Su_initial, 2);
    auto Sv2_initial = pow(Sv_initial, 2);
    auto Sw2_initial = pow(Sw_initial, 2);

    e_initial += -magnitude_(i) * ( a1_*Sw2_initial + a2_*pow2(Sw2_initial) +
                                    a3_*Sw_initial*Sv_initial*(3*Su2_initial-Sv2_initial) +
                                    a4_*(pow2(Su2_initial)+pow2(Sv2_initial)) +
                                    a5_*Su2_initial*Sv2_initial +
                                    a6_*(pow2(Su2_initial)-6*Su2_initial*Sv2_initial+pow2(Sv2_initial)) );

    /////////////////////////////////////////////////////
    /////////////  Final  ///////////////////////////////
    auto Su_final = dot(axis1_(i), spin_final);
    auto Sv_final = dot(axis2_(i), spin_final);
    auto Sw_final = dot(axis3_(i), spin_final);

    auto Su2_final = pow(Su_final, 2);
    auto Sv2_final = pow(Sv_final, 2);
    auto Sw2_final = pow(Sw_final, 2);

    e_final += -magnitude_(i) * ( a1_*Sw2_final + a2_*pow2(Sw2_final) +
                                    a3_*Sw_final*Sv_final*(3*Su2_final-Sv2_final) +
                                    a4_*(pow2(Su2_final)+pow2(Sv2_final)) +
                                    a5_*Su2_final*Sv2_final +
                                    a6_*(pow2(Su2_final)-6*Su2_final*Sv2_final+pow2(Sv2_final)) );

    // Return the result
    return e_final - e_initial;
}

void UniaxialGeneralisedHamiltonian::calculate_energies(double time) {
    for (auto i = 0; i < energy_.size(); ++i) {
        energy_(i) = calculate_energy(i, time);
    }
}

Vec3 UniaxialGeneralisedHamiltonian::calculate_field(const int i, double time) {
    Vec3 field = {0.0, 0.0, 0.0};
    Vec3 spin = {globals::s(i, 0), globals::s(i, 1), globals::s(i, 2)};

    // Calculate the dot product of the spin vector with the anisotropy axes
    auto Su = dot(axis1_(i), spin);
    auto Sv = dot(axis2_(i), spin);
    auto Sw = dot(axis3_(i), spin);

    for (auto j = 0; j < 3; ++j) {
        // Add Sw**2
        field[j] += a1_*2*Sw*axis3_(i)[j];

        // Add Sw**4
        field[j] += a2_*4*pow3(Sw)*axis3_(i)[j];

        // Add Sw * Sv * ( 3*Su**2 - Sv**2 )
        field[j] += a3_ * ( (6*Su*Sv*Sw) * axis1_(i)[j] + (Sv*(3*pow2(Su) - pow2(Sv) - 2*Sw*Sv )) * axis2_(i)[j] + (Sw*(3*pow2(Su) - pow2(Sv))) * axis3_(i)[j] );

        // Add Su**4 + Sv**4
        field[j] += a4_ * ( 4*axis1_(i)[j]*pow3(Su) + 4*axis2_(i)[j]*pow3(Sv) );

        // Add Su**2 * Sv**2
        field[j] += a5_ * ( 2*axis1_(i)[j]*Su*pow2(Sv) + 2*axis2_(i)[j]*Sv*pow2(Su) );

        // Add Su**4 - 6*Su**2*Sv**2 + Sv**4
        field[j] += a6_ * ( axis1_(i)[j]*Su*(4*pow2(Su) - 6*pow2(Sv)) +
                    axis2_(i)[j]*Sv*(4*pow2(Sv) - 6*pow2(Su)) );

        field[j] *= magnitude_(i);
    }
    return field;
}

void UniaxialGeneralisedHamiltonian::calculate_fields(double time) {
    for (auto i = 0; i < globals::num_spins; ++i) {
        const auto field = calculate_field(i, time);
        for (auto j = 0; j < 3; ++j) {
            field_(i, j) = field[j];
        }
    }
}