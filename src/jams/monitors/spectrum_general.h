//
// Created by Joe Barker on 2018/05/31.
//

#ifndef JAMS_SPECTRUM_GENERAL_H
#define JAMS_SPECTRUM_GENERAL_H

#include "jams/core/monitor.h"
#include <fstream>
#include <vector>
#include <complex>

#include "jblib/containers/array.h"

class SpectrumGeneralMonitor : public Monitor {
    friend class CudaSpectrumGeneralMonitor;

public:
    explicit SpectrumGeneralMonitor(const libconfig::Setting &settings);
    ~SpectrumGeneralMonitor() override;

    void update(Solver * solver) override;
    bool is_converged() override {return false;}
private:

    void apply_time_fourier_transform();

    std::ofstream outfile;

    unsigned num_samples_;
    unsigned padded_size_;
    double freq_delta_;
    unsigned time_point_counter_ = 0;

    unsigned num_q_ = 1;
    Vec3     qmax_  = {{0.0, 0.0, 0.0}};

    jblib::Array<std::complex<double>, 2> spin_data_;
};

#endif //JAMS_SPECTRUM_GENERAL_H
