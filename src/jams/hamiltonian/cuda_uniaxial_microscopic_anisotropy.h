//
// Created by Joe Barker on 2018/11/01.
//

#ifndef JAMS_CUDA_UNIAXIAL_MICROSCOPIC_ANISOTROPY_H
#define JAMS_CUDA_UNIAXIAL_MICROSCOPIC_ANISOTROPY_H

#include <cuda_runtime_api.h>

#include "jams/hamiltonian/uniaxial_microscopic_anisotropy.h"

class CudaUniaxialMicroscopicHamiltonian : public UniaxialMicroscopicHamiltonian {
public:
    CudaUniaxialMicroscopicHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaUniaxialMicroscopicHamiltonian() override = default;

    void   calculate_fields() override;
private:
    cudaStream_t dev_stream_ = nullptr;
    unsigned int dev_blocksize_;
    jblib::CudaArray<int, 1> dev_mca_order_;
    jblib::CudaArray<double, 1> dev_mca_value_;
};

#endif //JAMS_CUDA_UNIAXIAL_MICROSCOPIC_ANISOTROPY_H