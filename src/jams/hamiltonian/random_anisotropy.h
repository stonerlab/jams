//
// Created by Joe Barker on 2018/05/28.
//
#ifndef JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H
#define JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H

#include <jams/core/hamiltonian.h>
#include <jams/helpers/output.h>

#include <vector>

class RandomAnisotropyHamiltonian : public Hamiltonian {
    friend class CudaRandomAnisotropyHamiltonian;

public:
    RandomAnisotropyHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Vec<jams::Real, 3> calculate_field(int i, jams::Real time) override;
    jams::Real calculate_energy(int i, jams::Real time) override;

protected:
    jams::Real calculate_energy_for_spin(int i, const jams::Vec<double, 3> &spin, jams::Real time) override;

private:
    // write information about the random axes to outfile
    void output_anisotropy_axes(jams::output::TsvWriter& tsv);

    std::vector<jams::Vec<jams::Real, 3>> direction_; // local uniaxial anisotropy direction
    std::vector<jams::Real> magnitude_; // magnitude of local anisotropy
};

#endif  // JAMS_HAMILTONIAN_RANDOM_ANISOTROPY_H
