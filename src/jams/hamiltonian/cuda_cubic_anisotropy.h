//
// Created by Joe Barker on 2018/11/01.
//

#ifndef JAMS_CUDA_CUBIC_ANISOTROPY_H
#define JAMS_CUDA_CUBIC_ANISOTROPY_H

#include <jams/cuda/cuda_stream.h>
#include <jams/hamiltonian/cubic_anisotropy.h>

class CudaCubicAnisotropyHamiltonian : public CubicAnisotropyHamiltonian {
public:
    CudaCubicAnisotropyHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    jams::Real calculate_total_energy(jams::Real time) override;
    void   calculate_energies(jams::Real time) override;
    void   calculate_fields(jams::Real time) override;
private:
    unsigned int dev_blocksize_ = 64;
};

#endif //JAMS_CUDA_CUBIC_ANISOTROPY_H
