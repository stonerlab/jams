//
// Created by Sean Stansill [ll14s26s] on 30/03/2023.
//

#ifndef JAMS_CUDA_UNIAXIAL_GENERALISED_H
#define JAMS_CUDA_UNIAXIAL_GENERALISED_H

#include <jams/cuda/cuda_stream.h>
#include <jams/hamiltonian/uniaxial_generalised.h>

class CudaUniaxialGeneralisedHamiltonian : public UniaxialGeneralisedHamiltonian {
public:
    CudaUniaxialGeneralisedHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaUniaxialGeneralisedHamiltonian() override = default;

    double calculate_total_energy(double time) override;
    void   calculate_energies(double time) override;
    void   calculate_fields(double time) override;
private:

    CudaStream dev_stream_;
    unsigned int dev_blocksize_ = 64;
};

#endif //JAMS_CUDA_UNIAXIAL_GENERALISED_H
