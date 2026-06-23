//
// Created by Joseph Barker on 05/01/2026.
//

#ifndef JAMS_CUDA_LLG_RKMK4_H
#define JAMS_CUDA_LLG_RKMK4_H

#if HAS_CUDA

#include "jams/cuda/cuda_stream.h"
#include "jams/cuda/cuda_solver.h"
#include "jams/containers/multiarray.h"

class CUDALLGRKMK4Solver : public CudaSolver {
public:
    inline explicit CUDALLGRKMK4Solver(const libconfig::Setting &settings) {
        initialize(settings);
    }

    void initialize(const libconfig::Setting& settings) override;
    void run() override;

    std::string name() const override { return "llg-rkmk4-gpu"; }
private:
    template <typename GyroParam, typename AlphaParam, typename FieldScaleParam>
    void run_with_parameters(GyroParam gyro, AlphaParam alpha, FieldScaleParam field_scale);

    CudaStream dev_stream_;
    jams::MultiArray<jams::Real, 1> gyro_eff_;
    bool gyro_eff_is_uniform_ = false;
    jams::Real gyro_eff_uniform_value_ = jams::Real{0.0};
    bool alpha_is_uniform_ = false;
    jams::Real alpha_uniform_value_ = jams::Real{0.0};
    bool mus_is_uniform_ = false;
    jams::Real mus_uniform_inv_value_ = jams::Real{0.0};
    jams::MultiArray<double, 2> s_init_;
    jams::MultiArray<double, 2> k1_;
    jams::MultiArray<double, 2> k2_;
    jams::MultiArray<double, 2> k3_;
};

#endif

#endif //JAMS_CUDA_LLG_RKMK4_H
