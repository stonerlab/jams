//
// Created by Sean Stansill [ll14s26s] on 28/10/2019.
//
#include "jams/core/globals.h"
#include "jams/core/solver.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/maths.h"
#include "jams/helpers/consts.h"
#include "jams/helpers/error.h"
#include "jams/hamiltonian/cubic_anisotropy.h"

using libconfig::Setting;
using std::vector;
using std::string;
using std::runtime_error;
// Settings should look like:
// {
//    module = "Cubic";
//    K1 = (1e-24, 2e-24); for two magnetic materials with standard cubic axes of [1,0,0], [0,1,0] and [0,0,1]
// }
//
// {
//    module = "cubic";
//    K1 = ((1e-24, [1, 0, 0], [0, 1, 0], [0, 0, 1]),
//          (2e-24, [1, 0, 0], [0, 1, 0], [0, 0, 1])); specifying the different cubic axes for two materials. Vectors don't have to be normalized as this is done in the code
//          Can add in a check that two are orthogonal similar to the mumax implementation
// }

struct AnisotropySetting_cube {
    unsigned power;
    double   energy;
    Vec3     axis1;
    Vec3     axis2;
    Vec3     axis3;
};

unsigned anisotropy_power_from_name_cube(const string name) {
    if (name == "K1") return 1; // This is outputted correctly
    if (name == "K2") return 2;
    throw runtime_error("Unsupported anisotropy: " + name);
}
//Need to remove use of a defined axis. This might be able to become the distortion term for Mai
AnisotropySetting read_anisotropy_setting_cube(Setting &setting) {
    if (setting.isList()) {
        Vec3 axis1 = {setting[1][0], setting[1][1], setting[1][2]};
        Vec3 axis2 = {setting[2][0], setting[2][1], setting[2][2]};
        Vec3 axis3 = {setting[3][0], setting[3][1], setting[3][2]};
        return AnisotropySetting_cube{anisotropy_power_from_name_cube(setting.getParent().getName()), setting[0], normalize(axis1), normalize(axis2), normalize(axis3)};
    }
    if (setting.isScalar()) {
        return AnisotropySetting_cube{anisotropy_power_from_name_cube(setting.getParent().getName()), setting, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    }
    throw runtime_error("Incorrectly formatted anisotropy setting");
}

vector<vector<AnisotropySetting_cube>> read_all_anisotropy_settings_cube(const Setting &settings) {
    vector<vector<AnisotropySetting_cube>> anisotropies_cube(lattice->num_materials());
    auto anisotropy_names_cube = {"K1", "K2"};
    for (const auto name : anisotropy_names_cube) {
        if (settings.exists(name)) {
            if (settings[name].getLength() != lattice->num_materials()) {
                throw runtime_error("CubicHamiltonian: " + string(name) + "  must be specified for every material");
            }

            for (auto i = 0; i < settings[name].getLength(); ++i) {
                anisotropies_cube[i].push_back(read_anisotropy_setting_cube(settings[name][i]));
            }
        }
    }
    // the array indicies are (type, power)
    return anisotropies;
}

CubicHamiltonian::CubicHamiltonian(const Setting &settings, const unsigned int num_spins)
        : Hamiltonian(settings, num_spins) {

    // check if the old format is being used
    //if ((settings.exists("d2z") || settings.exists("d4z") || settings.exists("d6z"))) {
    //    jams_die(
    //            "CubicHamiltonian: anisotropy should only be specified in terms of K1, K2, K3 maybe you want UniaxialCoefficientHamiltonian?");
    //}

    auto anisotropies_cube = read_all_anisotropy_settings_cube(settings);

    for (auto type = 0; type < lattice->num_materials(); ++type) {
        std::cout << "  " << lattice->material_name(type) << ":\n";
        for (const auto& ani : anisotropies_cube[type]) {
            std::cout << "    " << ani.axis1 << "  " << ani.axis2 << "  " << ani.axis3 << "  " << ani.power << "  " << ani.energy << "\n";
        }
    }

    num_coefficients_ = anisotropies_cube[0].size();

    power_.resize(num_spins, anisotropies_cube[0].size());
    axis_.resize(num_spins, anisotropies_cube[0].size(), 3, 3); //Added in another dimension of size 3
    magnitude_.resize(num_spins, anisotropies_cube[0].size());

    for (int i = 0; i < globals::num_spins; ++i) {
        auto type = lattice->atom_material_id(i);
        for (auto j = 0; j < anisotropies_cube[type].size(); ++j) {
            power_(i, j) = anisotropies_cube[type][j].power;
            magnitude_(i, j) = anisotropies_cube[type][j].energy * input_unit_conversion_;
            for (auto k : {0,1,2}) {
                axis_(i, j, k, 0) = anisotropies_cube[type][j].axis1[k];
                axis_(i, j, k, 1) = anisotropies_cube[type][j].axis2[k];
                axis_(i, j, k, 2) = anisotropies_cube[type][j].axis3[k];
            }
        }
    }
}


double CubicHamiltonian::calculate_total_energy_cube() {
    double e_total = 0.0;
    for (int i = 0; i < energy_.size(); ++i) {
        e_total += calculate_one_spin_energy_cube(i);
    }
    return e_total;
}

double CubicHamiltonian::calculate_one_spin_energy_cube(const int i) {
    using namespace globals;
    double energy = 0.0;

    for (auto n = 0; n < num_coefficients_; ++n) {
        auto dot1 = s(i,0);
        auto dot2 = s(i,1);
        auto dot3 = s(i,2);
        /*auto dot1 = axis_(i,n,0,0)*s(i,0) + axis_(i,n,1,0)*s(i,1) + axis_(i,n,2,0)*s(i,2);
        auto dot2 = axis_(i,n,0,1)*s(i,0) + axis_(i,n,1,1)*s(i,1) + axis_(i,n,2,1)*s(i,2);
        auto dot3 = axis_(i,n,0,2)*s(i,0) + axis_(i,n,1,2)*s(i,1) + axis_(i,n,2,2)*s(i,2);*/

//        if(power_(i, n) == 1) {
            energy += -magnitude_(i,n) * (pow(dot1,2)*pow(dot2,2) + pow(dot2,2)*pow(dot3,2) + pow(dot3,2)*pow(dot1,2));
//        }
//        if(power_(i, n) == 2){
//            energy += -magnitude_(i,n) * (pow(dot1,2)*pow(dot2,2)*pow(dot3,2));
//        }
    }

    return energy;
}

double CubicHamiltonian::calculate_one_spin_energy_difference_cube(const int i, const Vec3 &spin_initial,
                                                              const Vec3 &spin_final) {
    double e_initial = 0.0;
    double e_final = 0.0;

    for (auto n = 0; n < num_coefficients_; ++n) {
        auto dot1 = axis_(i,n,0,0)*spin_initial[0] + axis_(i,n,1,0)*spin_initial[1] + axis_(i,n,2,0)*spin_initial[2];
        auto dot2 = axis_(i,n,0,1)*spin_initial[0] + axis_(i,n,1,1)*spin_initial[1] + axis_(i,n,2,1)*spin_initial[2];
        auto dot3 = axis_(i,n,0,2)*spin_initial[0] + axis_(i,n,1,2)*spin_initial[1] + axis_(i,n,2,2)*spin_initial[2];

//        if(power_(i, n) == 1) {  // Alternatively use if (n == 0){} else if (n == 1){} and remove power_ from cubic anisotropy. Note: Don't think this will work from the declaration of num_coefficients_
            e_initial = -magnitude_(i,n) * (pow(dot1,2)*pow(dot2,2) + pow(dot2,2)*pow(dot3,2) + pow(dot3,2)*pow(dot1,2));
//        }
//        if(power_(i, n) == 2){
//            e_initial = -magnitude_(i,n) * (pow(dot1,2)*pow(dot2,2)*pow(dot3,2));
//        }
    }

    for (auto n = 0; n < num_coefficients_; ++n) {
        auto dot1 = axis_(i,n,0,0)*spin_final[0] + axis_(i,n,1,0)*spin_final[1] + axis_(i,n,2,0)*spin_final[2];
        auto dot2 = axis_(i,n,0,1)*spin_final[0] + axis_(i,n,1,1)*spin_final[1] + axis_(i,n,2,1)*spin_final[2];
        auto dot3 = axis_(i,n,0,2)*spin_final[0] + axis_(i,n,1,2)*spin_final[1] + axis_(i,n,2,2)*spin_final[2];

//        if(power_(i, n) == 1) {  // Alternatively use if (n == 0){} else if (n == 1){} and remove power_ from cubic anisotropy. Note: Don't think this will work from the declaration of num_coefficients_
            e_final = -magnitude_(i,n) * (pow(dot1,2)*pow(dot2,2) + pow(dot2,2)*pow(dot3,2) + pow(dot3,2)*pow(dot1,2));
//        }
//        if(power_(i, n) == 2){
//            e_final = -magnitude_(i,n) * (pow(dot1,2)*pow(dot2,2)*pow(dot3,2));
//        }
    }

    return e_final - e_initial;
}

void CubicHamiltonian::calculate_energies_cube() {
    for (int i = 0; i < energy_.size(); ++i) {
        energy_(i) = calculate_one_spin_energy_cube(i);
    }
}

void CubicHamiltonian::calculate_one_spin_field_cube(const int i, double local_field[3]) {
    using namespace globals;
    local_field[0] = 0.0;
    local_field[1] = 0.0;
    local_field[2] = 0.0;

    for (auto n = 0; n < num_coefficients_; ++n) {
//        if (power_(i, n) == 1) {
            auto dot1 = axis_(i, n, 0, 0) * s(i, 0) + axis_(i, n, 1, 0) * s(i, 1) + axis_(i, n, 2, 0) * s(i, 2);
            auto dot2 = axis_(i, n, 0, 1) * s(i, 0) + axis_(i, n, 1, 1) * s(i, 1) + axis_(i, n, 2, 1) * s(i, 2);
            auto dot3 = axis_(i, n, 0, 2) * s(i, 0) + axis_(i, n, 1, 2) * s(i, 1) + axis_(i, n, 2, 2) * s(i, 2);
            for (auto j = 0; j < 3; ++j) {
                local_field[j] += 2 * magnitude_(i, n) * ((pow(dot2, 2) + pow(dot3, 2)) * (dot1 * (axis_(i, n, j, 0) +
                                                                                                   axis_(i, n, j, 1) +
                                                                                                   axis_(i, n, j, 2))));
                local_field[j] += 2 * magnitude_(i, n) * ((pow(dot1, 2) + pow(dot3, 2)) * (dot2 * (axis_(i, n, j, 0) +
                                                                                                   axis_(i, n, j, 1) +
                                                                                                   axis_(i, n, j, 2))));
                local_field[j] += 2 * magnitude_(i, n) * ((pow(dot1, 2) + pow(dot2, 2)) * (dot3 * (axis_(i, n, j, 0) +
                                                                                                   axis_(i, n, j, 1) +
                                                                                                   axis_(i, n, j, 2))));
            }
        }

/*        if (power_(i, n) == 2) {
            auto dot1 = axis_(i, n, 0, 0) * s(i, 0) + axis_(i, n, 1, 0) * s(i, 1) + axis_(i, n, 2, 0) * s(i, 2);
            auto dot2 = axis_(i, n, 0, 1) * s(i, 0) + axis_(i, n, 1, 1) * s(i, 1) + axis_(i, n, 2, 1) * s(i, 2);
            auto dot3 = axis_(i, n, 0, 2) * s(i, 0) + axis_(i, n, 1, 2) * s(i, 1) + axis_(i, n, 2, 2) * s(i, 2);
            for (auto j = 0; j < 3; ++j) {
                local_field[j] += 2 * magnitude_(i, n) * ((dot1 * pow(dot2, 2) * pow(dot3, 2) *
                                                           (axis_(i, n, j, 0) + axis_(i, n, j, 1) +
                                                            axis_(i, n, j, 2))));
                local_field[j] += 2 * magnitude_(i, n) * ((dot2 * pow(dot1, 2) * pow(dot3, 2) *
                                                           (axis_(i, n, j, 0) + axis_(i, n, j, 1) +
                                                            axis_(i, n, j, 2))));
                local_field[j] += 2 * magnitude_(i, n) * ((dot3 * pow(dot2, 2) * pow(dot1, 2) *
                                                           (axis_(i, n, j, 0) + axis_(i, n, j, 1) +
                                                            axis_(i, n, j, 2))));
            }
        }
    }*/
}

void CubicHamiltonian::calculate_fields_cube() {
    using namespace globals;
    field_.zero();

    for (auto i = 0; i < num_spins; ++i) {
        for (auto n = 0; n < num_coefficients_; ++n) {
//            if (power_(i, n) == 1) {
                auto dot1 = axis_(i, n, 0, 0) * s(i, 0) + axis_(i, n, 1, 0) * s(i, 1) + axis_(i, n, 2, 0) * s(i, 2);
                auto dot2 = axis_(i, n, 0, 1) * s(i, 0) + axis_(i, n, 1, 1) * s(i, 1) + axis_(i, n, 2, 1) * s(i, 2);
                auto dot3 = axis_(i, n, 0, 2) * s(i, 0) + axis_(i, n, 1, 2) * s(i, 1) + axis_(i, n, 2, 2) * s(i, 2);
                for (auto j = 0; j < 3; ++j) {
                    field_(i,j) += 2 * magnitude_(i, n) * ((pow(dot2, 2) + pow(dot3, 2)) * (dot1 * (axis_(i, n, j, 0) +
                                                                                                       axis_(i, n, j, 1) +
                                                                                                       axis_(i, n, j, 2))));
                    field_(i,j) += 2 * magnitude_(i, n) * ((pow(dot1, 2) + pow(dot3, 2)) * (dot2 * (axis_(i, n, j, 0) +
                                                                                                       axis_(i, n, j, 1) +
                                                                                                       axis_(i, n, j, 2))));
                    field_(i,j) += 2 * magnitude_(i, n) * ((pow(dot1, 2) + pow(dot2, 2)) * (dot3 * (axis_(i, n, j, 0) +
                                                                                                       axis_(i, n, j, 1) +
                                                                                                       axis_(i, n, j, 2))));
                }
//            }

/*            if (power_(i, n) == 2) {
                auto dot1 = axis_(i, n, 0, 0) * s(i, 0) + axis_(i, n, 1, 0) * s(i, 1) + axis_(i, n, 2, 0) * s(i, 2);
                auto dot2 = axis_(i, n, 0, 1) * s(i, 0) + axis_(i, n, 1, 1) * s(i, 1) + axis_(i, n, 2, 1) * s(i, 2);
                auto dot3 = axis_(i, n, 0, 2) * s(i, 0) + axis_(i, n, 1, 2) * s(i, 1) + axis_(i, n, 2, 2) * s(i, 2);
                for (auto j = 0; j < 3; ++j) {
                    field_(i,j) += 2 * magnitude_(i, n) * ((dot1 * pow(dot2, 2) * pow(dot3, 2) *
                                                               (axis_(i, n, j, 0) + axis_(i, n, j, 1) +
                                                                axis_(i, n, j, 2))));
                    field_(i,j) += 2 * magnitude_(i, n) * ((dot2 * pow(dot1, 2) * pow(dot3, 2) *
                                                               (axis_(i, n, j, 0) + axis_(i, n, j, 1) +
                                                                axis_(i, n, j, 2))));
                    field_(i,j) += 2 * magnitude_(i, n) * ((dot3 * pow(dot2, 2) * pow(dot1, 2) *
                                                               (axis_(i, n, j, 0) + axis_(i, n, j, 1) +
                                                                axis_(i, n, j, 2))));
                }
            }*/
        }
    }
}