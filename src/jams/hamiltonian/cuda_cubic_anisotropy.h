//
// Created by Joe Barker on 2018/11/01.
//

#ifndef JAMS_CUDA_CUBIC_ANISOTROPY_H
#define JAMS_CUDA_CUBIC_ANISOTROPY_H

#include <cuda_runtime_api.h>

#include "jams/cuda/cuda_stream.h"
#include "jams/hamiltonian/cubic_anisotropy.h"

class CudaCubicHamiltonian : public CubicHamiltonian {
public:
    CudaCubicHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaCubicHamiltonian() override = default;

    void   calculate_fields(double time) override;
private:

    CudaStream dev_stream_;
    unsigned int dev_blocksize_ = 64;
};

#endif //JAMS_CUDA_CUBIC_ANISOTROPY_H
