// cuda_rk4_base.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_RK4_BASE
#define INCLUDED_JAMS_CUDA_RK4_BASE

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CudaRK4BaseSolver : public CudaSolver {
public:
    CudaRK4BaseSolver() = default;
    ~CudaRK4BaseSolver() override = default;

    inline explicit CudaRK4BaseSolver(const libconfig::Setting &settings) {
      initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

protected:
    // Function called before each integration step
    virtual void pre_step(jams::MultiArray<double, 2>& spins) {};

    // Function which is being integrated with the intermediate 'k' being written to
    virtual void function_kernel(jams::MultiArray<double, 2>& spins, jams::MultiArray<double, 2>& k) = 0;

    // Function called after each integration step
    virtual void post_step(jams::MultiArray<double, 2>& spins) {};

    CudaStream dev_stream_;

private:
    jams::MultiArray<double, 2> s_old_;
    jams::MultiArray<double, 2> k1_;
    jams::MultiArray<double, 2> k2_;
    jams::MultiArray<double, 2> k3_;
    jams::MultiArray<double, 2> k4_;
};

#endif

#endif
// ----------------------------- END-OF-FILE ----------------------------------