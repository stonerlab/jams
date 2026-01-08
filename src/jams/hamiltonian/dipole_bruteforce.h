//
// Created by Joseph Barker on 2020-04-26.
//

#ifndef JAMS_DIPOLE_BRUTEFORCE_H
#define JAMS_DIPOLE_BRUTEFORCE_H

#include <jams/core/hamiltonian.h>
#include <jams/core/types.h>

class DipoleBruteforceHamiltonian : public Hamiltonian {
public:
    DipoleBruteforceHamiltonian(const libconfig::Setting &settings, unsigned int size);

    Vec3R calculate_field(int i, jams::Real time) override;

    jams::Real calculate_energy(int i, jams::Real time) override;

    jams::Real calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, jams::Real time) override;

private:
    std::vector<Vec3>   frac_positions_;
    Mat3R supercell_matrix_;
    jams::Real r_cutoff_;
};

#endif //JAMS_DIPOLE_BRUTEFORCE_H
