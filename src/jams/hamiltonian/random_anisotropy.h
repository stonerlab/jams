//
// Created by Joe Barker on 2018/05/28.
//
#ifndef JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H
#define JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H

#include <vector>

#include <libconfig.h++>

#include "jams/core/hamiltonian.h"

class RandomAnisotropyHamiltonian : public Hamiltonian {
    friend class CudaRandomAnisotropyHamiltonian;

public:
    RandomAnisotropyHamiltonian(const libconfig::Setting &settings, unsigned int size);

    double calculate_total_energy() override;

    void calculate_energies() override;

    void calculate_fields() override;

    Vec3 calculate_one_spin_field(int i) override;

    double calculate_one_spin_energy(int i) override;

    double calculate_one_spin_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final) override;

private:
    // write information about the random axes to outfile
    void output_anisotropy_axes(std::ofstream &outfile);

    std::vector<Vec3> direction_; // local uniaxial anisotropy direction
    std::vector<double> magnitude_; // magnitude of local anisotropy
};

#endif  // JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H