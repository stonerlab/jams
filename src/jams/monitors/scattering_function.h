//
// Created by Joe Barker on 2018/05/31.
//

#ifndef JAMS_SCATTERING_FUNCTION_H
#define JAMS_SCATTERING_FUNCTION_H

#include "jams/core/monitor.h"
#include <fstream>
#include <vector>
#include <complex>

class ScatteringFunctionMonitor : public Monitor {
public:
    ScatteringFunctionMonitor(const libconfig::Setting &settings);
    ~ScatteringFunctionMonitor();

    void update(Solver * solver);
    bool is_converged();
private:
    std::ofstream outfile;

    unsigned num_kpoints_;
    unsigned num_samples_;
    double t_sample_;

    std::vector<std::vector<double>> sx_;
    std::vector<std::vector<double>> sy_;
    std::vector<std::vector<double>> sz_;

    std::vector<double> time_correlation(unsigned int i, unsigned int j, unsigned subsample);
};

#endif //JAMS_SCATTERING_FUNCTION_H
