//
// Created by Joseph Barker on 2018-11-22.
//

#ifndef JAMS_CUDA_SPECTRUM_GENERAL_H
#define JAMS_CUDA_SPECTRUM_GENERAL_H

#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include <jblib/containers/cuda_array/cuda_array_template.h>
#include "jams/monitors/spectrum_general.h"

class CudaSpectrumGeneralMonitor : public SpectrumGeneralMonitor {
public:
    explicit CudaSpectrumGeneralMonitor(const libconfig::Setting &settings);
    using SpectrumGeneralMonitor::update;
    void post_process() override {};

    ~CudaSpectrumGeneralMonitor() override;
private:
};

#endif //JAMS_CUDA_SPECTRUM_GENERAL_H
