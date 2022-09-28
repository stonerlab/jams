//
// Created by Joe Barker on 2018/11/01.
//

#ifndef JAMS_CUDA_UNIAXIAL_ANISOTROPY_H
#define JAMS_CUDA_UNIAXIAL_ANISOTROPY_H

#include <cuda_runtime_api.h>

#include "jams/cuda/cuda_stream.h"
#include "jams/hamiltonian/uniaxial_anisotropy.h"

class CudaUniaxialHamiltonian : public UniaxialHamiltonian {
public:
    CudaUniaxialHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaUniaxialHamiltonian() override = default;

    double calculate_total_energy() override;
    void   calculate_energies() override;
    void   calculate_fields() override;
    void calculate_internal_energy_differences();
    double calculate_total_internal_energy_difference() override;
    void calculate_entropies();
    double calculate_entropy(int i) override;
    double calculate_total_entropy() override;

private:

    CudaStream dev_stream_;
    unsigned int dev_blocksize_ = 64;
    jams::MultiArray<double, 1> helicity_internal_;
    jams::MultiArray<double, 1> helicity_entropy_;
};

#endif //JAMS_CUDA_UNIAXIAL_ANISOTROPY_H
