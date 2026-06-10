//
// Created by Joseph Barker on 17/06/2024.
//

#ifndef JAMS_CUDA_CRYSTAL_FIELD_H
#define JAMS_CUDA_CRYSTAL_FIELD_H

#include <jams/hamiltonian/crystal_field.h>
#include <jams/cuda/cuda_stream.h>

class CudaCrystalFieldHamiltonian : public CrystalFieldHamiltonian {
public:
    CudaCrystalFieldHamiltonian(const libconfig::Setting &settings, unsigned int size);

    jams::Real calculate_total_energy(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;

    void calculate_energies(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;

    void calculate_fields(jams::Real time, const jams::MultiArray<jams::Real, 2>& spins) override;
};

#endif //JAMS_CUDA_CRYSTAL_FIELD_H
