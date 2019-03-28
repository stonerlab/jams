//
// Created by Joseph Barker on 2019-03-12.
//

#ifndef JAMS_SPIN_CORRELATION_H
#define JAMS_SPIN_CORRELATION_H

#include <complex>

#include "jblib/containers/array.h"
#include "jams/core/monitor.h"

class SpinCorrelationMonitor : public Monitor {
public:
    explicit SpinCorrelationMonitor(const libconfig::Setting &settings);

    void update(Solver * solver) override;
    void post_process() override;
    bool is_converged() override {return false;}
private:
    struct histo {
        double   Szz = 0.0;
        double   Szz_sq = 0.0;
        std::complex<double>  S_plus_minus = {0.0, 0.0};
        unsigned count = 0;
    };

    struct float_compare {
        bool operator()(const double &x, const double &y) const {
          return (x < y) && std::abs( x - y ) > 1e-5;
        };
    };


    unsigned num_samples_;
    unsigned time_point_counter_ = 0;

    jblib::Array<Vec3f, 2> spin_data_;   // index is spin, time. This order gives a large speedup
};

#endif //JAMS_SPIN_CORRELATION_H
