//
// Created by Joe Barker on 2018/11/01.
//

#ifndef JAMS_CUDA_UNIAXIAL_ANISOTROPY_H
#define JAMS_CUDA_UNIAXIAL_ANISOTROPY_H

#include <jams/cuda/cuda_stream.h>
#include <jams/hamiltonian/uniaxial_anisotropy.h>

class CudaUniaxialAnisotropyHamiltonian : public UniaxialAnisotropyHamiltonian {
public:
    CudaUniaxialAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaUniaxialAnisotropyHamiltonian() override = default;

    double calculate_total_energy(double time) override;
    void   calculate_energies(double time) override;
    void   calculate_fields(double time) override;
private:
    unsigned int dev_blocksize_ = 64;
};

#endif //JAMS_CUDA_UNIAXIAL_ANISOTROPY_H
