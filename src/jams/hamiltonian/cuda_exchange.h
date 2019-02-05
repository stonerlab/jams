//
// Created by Joe Barker on 2018/10/31.
//

#ifndef JAMS_CUDA_EXCHANGE_H
#define JAMS_CUDA_EXCHANGE_H

#include <cusparse.h>

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_defs.h"
#include "jams/cuda/cuda_sparsematrix.h"
#include "jams/hamiltonian/exchange.h"

class CudaExchangeHamiltonian : public ExchangeHamiltonian {
public:
    CudaExchangeHamiltonian(const libconfig::Setting &settings, const unsigned int size);

    double calculate_total_energy();
    void calculate_fields();

private:

#if HAS_CUSPARSE_MIXED_PREC
  CudaSparseMatrixCSR<float> dev_csr_interaction_matrix_;
  size_t dev_csr_buffer_size_;
  void*  dev_csr_buffer_;
#else
  CudaSparseMatrixCSR<double> dev_csr_interaction_matrix_;
#endif
    cusparseHandle_t   cusparse_handle_;
    CudaStream dev_stream_;
};

#endif //JAMS_CUDA_EXCHANGE_H
