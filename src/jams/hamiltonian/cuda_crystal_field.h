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

    double calculate_total_energy(double time) override;

    void calculate_energies(double time) override;

    void calculate_fields(double time) override;
};

#endif //JAMS_CUDA_CRYSTAL_FIELD_H
