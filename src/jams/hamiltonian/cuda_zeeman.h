//
// Created by Joe Barker on 2018/10/31.
//

#ifndef JAMS_CUDA_ZEEMAN_H
#define JAMS_CUDA_ZEEMAN_H

#include <jams/hamiltonian/zeeman.h>

class CudaZeemanHamiltonian : public ZeemanHamiltonian {
public:
    CudaZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaZeemanHamiltonian();

    void calculate_fields(double time);

private:
    cudaStream_t dev_stream_ = nullptr;
};
#endif //JAMS_CUDA_ZEEMAN_H
