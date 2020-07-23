//
// Created by Joseph Barker on 2020-04-26.
//

#ifndef JAMS_DIPOLE_BRUTEFORCE_H
#define JAMS_DIPOLE_BRUTEFORCE_H

#include "jams/helpers/maths.h"
#include "jams/core/hamiltonian.h"

class DipoleBruteforceHamiltonian : public Hamiltonian {
public:
    DipoleBruteforceHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_field(int i) override;

    double calculate_energy(int i) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:
    std::vector<Vec3>   frac_positions_;
    Mat3 supercell_matrix_;
    double r_cutoff_;
};

#endif //JAMS_DIPOLE_BRUTEFORCE_H
