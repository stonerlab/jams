// cuda_gse_rk4.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_GSE_RK4
#define INCLUDED_JAMS_CUDA_GSE_RK4
/// @brief:
///
/// @details: This component...
///
/// Usage
/// -----

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/solvers/cuda_rk4_base.h"
#include "jams/containers/multiarray.h"

class CUDAGSERK4Solver : public CudaRK4BaseSolver {
public:
    inline explicit CUDAGSERK4Solver(const libconfig::Setting &settings) : CudaRK4BaseSolver(settings) {};

    std::string name() const override { return "gse-rk4-gpu"; }

    void function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k) override;
};

#endif

#endif
// ----------------------------- END-OF-FILE ----------------------------------