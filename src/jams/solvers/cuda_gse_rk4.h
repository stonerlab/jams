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
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDAGSERK4Solver : public CudaSolver {
public:
    CUDAGSERK4Solver() = default;
    ~CUDAGSERK4Solver() override = default;

    inline explicit CUDAGSERK4Solver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "gse-rk4-gpu"; }

private:
    CudaStream dev_stream_;

    jams::MultiArray<double, 2> s_old_;
    jams::MultiArray<double, 2> k1_;
    jams::MultiArray<double, 2> k2_;
    jams::MultiArray<double, 2> k3_;
    jams::MultiArray<double, 2> k4_;
};

#endif

#endif
// ----------------------------- END-OF-FILE ----------------------------------