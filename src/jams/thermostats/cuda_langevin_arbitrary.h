#ifndef JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_H
#define JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_H

#if HAS_CUDA

#include <curand.h>
#include <fstream>
#include <mutex>

#include "jams/core/thermostat.h"

class CudaLangevinArbitraryThermostat : public Thermostat {
public:
    CudaLangevinArbitraryThermostat(const double &temperature, const double &sigma, const int num_spins);
    ~CudaLangevinArbitraryThermostat();

    void update();

    // override the base class implementation
    const double* device_data() { return noise_.device_data(); }

private:
    std::ofstream debug_file_;
    bool debug_;
    bool do_zero_point_ = false;

    int    num_freq_;
    int    num_trunc_;
    double max_omega_;
    double delta_omega_;
    double delta_t_;
    double filter_temperature_;

    jams::MultiArray<double, 1> filter_;
    jams::MultiArray<double, 1> white_noise_;

    cudaStream_t                dev_stream_ = nullptr;
    cudaStream_t                dev_curand_stream_ = nullptr;
};

#endif  // CUDA
#endif  // JAMS_CUDA_THERMOSTAT_LANGEVIN_ARBITRARY_H
