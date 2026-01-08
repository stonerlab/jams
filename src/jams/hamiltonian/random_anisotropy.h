//
// Created by Joe Barker on 2018/05/28.
//
#ifndef JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H
#define JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H

#include <jams/core/hamiltonian.h>

#include <iosfwd>
#include <vector>

class RandomAnisotropyHamiltonian : public Hamiltonian {
    friend class CudaRandomAnisotropyHamiltonian;

public:
    RandomAnisotropyHamiltonian(const libconfig::Setting &settings, unsigned int size);

    Vec3R calculate_field(int i, jams::Real time) override;
    jams::Real calculate_energy(int i, jams::Real time) override;

private:
    // write information about the random axes to outfile
    void output_anisotropy_axes(std::ofstream &outfile);

    std::vector<Vec3R> direction_; // local uniaxial anisotropy direction
    std::vector<jams::Real> magnitude_; // magnitude of local anisotropy
};

#endif  // JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H