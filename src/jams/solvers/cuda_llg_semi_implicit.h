#ifndef JAMS_SOLVER_CUDA_LLG_SemiImplict_H
#define JAMS_SOLVER_CUDA_LLG_SemiImplict_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLGSemiImplictSolver : public CudaSolver {
public:
    inline explicit CUDALLGSemiImplictSolver(const libconfig::Setting &settings) {
        initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-simp-gpu"; }
private:
    template <typename GyroParam, typename AlphaParam, typename DtGyroMuParam>
    void run_with_parameters(GyroParam gyro, AlphaParam alpha, DtGyroMuParam dt_gyro_mu);

    jams::MultiArray<double, 2> s_init_;
    jams::MultiArray<jams::Real, 1> gyro_eff_;
    jams::MultiArray<jams::Real, 1> dt_gyro_mu_;
    bool gyro_eff_is_uniform_ = false;
    jams::Real gyro_eff_uniform_value_ = jams::Real{0.0};
    bool alpha_is_uniform_ = false;
    jams::Real alpha_uniform_value_ = jams::Real{0.0};
    bool dt_gyro_mu_is_uniform_ = false;
    jams::Real dt_gyro_mu_uniform_value_ = jams::Real{0.0};
};

#endif

#endif // JAMS_SOLVER_CUDA_LLG_SemiImplict_H
