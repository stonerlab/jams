//
// Created by Joe Barker on 2018/05/31.
//

#ifndef JAMS_SPECTRUM_GENERAL_H
#define JAMS_SPECTRUM_GENERAL_H

#include <jams/core/monitor.h>
#include <jams/containers/multiarray.h>

#include <fstream>
#include <vector>
#include <complex>

class SpectrumGeneralMonitor : public Monitor {
    friend class CudaSpectrumGeneralMonitor;

public:
    explicit SpectrumGeneralMonitor(const libconfig::Setting &settings);
    ~SpectrumGeneralMonitor() override;

    void update(Solver& solver) override;
    void post_process() override {};
private:

    void apply_time_fourier_transform();

    std::ofstream outfile;

    unsigned num_samples_;
    unsigned padded_size_;
    double freq_delta_;
    unsigned time_point_counter_ = 0;

    unsigned num_qvectors_ = 1;
    unsigned num_qpoints_ = 1;
    double   qmax_ = 0.0;

    jams::MultiArray<std::complex<double>, 2> spin_data_;
};

#endif //JAMS_SPECTRUM_GENERAL_H
