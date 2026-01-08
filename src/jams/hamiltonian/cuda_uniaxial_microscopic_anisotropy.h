//
// Created by Joe Barker on 2018/11/01.
//

#ifndef JAMS_CUDA_UNIAXIAL_MICROSCOPIC_ANISOTROPY_H
#define JAMS_CUDA_UNIAXIAL_MICROSCOPIC_ANISOTROPY_H

#include <jams/hamiltonian/uniaxial_microscopic_anisotropy.h>

#include <cuda_runtime_api.h>

class CudaUniaxialMicroscopicAnisotropyHamiltonian : public UniaxialMicroscopicAnisotropyHamiltonian {
public:
    CudaUniaxialMicroscopicAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaUniaxialMicroscopicAnisotropyHamiltonian() override = default;

    void   calculate_fields(double time) override;
private:
    unsigned int dev_blocksize_;
};

#endif //JAMS_CUDA_UNIAXIAL_MICROSCOPIC_ANISOTROPY_H
