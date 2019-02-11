//
// Created by Joe Barker on 2018/10/31.
//

#ifndef JAMS_CUDA_EXCHANGE_H
#define JAMS_CUDA_EXCHANGE_H

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_common.h"
#include "jams/cuda/cuda_sparse_interaction_matrix.h"
#include "jams/hamiltonian/exchange.h"

class CudaExchangeHamiltonian : public ExchangeHamiltonian {
public:
    CudaExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    double calculate_total_energy();
    void calculate_fields();

private:

    CudaSparseInteractionMatrix<double> dev_interaction_matrix_;
    CudaStream dev_stream_;
};

#endif //JAMS_CUDA_EXCHANGE_H
