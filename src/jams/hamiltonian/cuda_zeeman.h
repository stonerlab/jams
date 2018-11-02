//
// Created by Joe Barker on 2018/10/31.
//

#ifndef JAMS_CUDA_ZEEMAN_H
#define JAMS_CUDA_ZEEMAN_H

#include "jams/hamiltonian/zeeman.h"

class CudaZeemanHamiltonian : public ZeemanHamiltonian {
public:
    CudaZeemanHamiltonian(const libconfig::Setting &settings, const unsigned int size);
    ~CudaZeemanHamiltonian();

    void calculate_fields();

private:
    cudaStream_t dev_stream_ = nullptr;
        jblib::CudaArray<double, 1> dev_dc_local_field_;
        jblib::CudaArray<double, 1> dev_ac_local_field_;
        jblib::CudaArray<double, 1> dev_ac_local_frequency_;
};
#endif //JAMS_CUDA_ZEEMAN_H
