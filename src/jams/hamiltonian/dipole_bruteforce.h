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

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;

    Vec3 calculate_field(int i, double time) override;

    double calculate_energy(int i, double time) override;

    double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) override;

private:
    std::vector<Vec3>   frac_positions_;
    Mat3 supercell_matrix_;
    double r_cutoff_;
};

#endif //JAMS_DIPOLE_BRUTEFORCE_H
