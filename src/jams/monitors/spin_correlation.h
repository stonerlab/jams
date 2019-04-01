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
    template <typename T>
    struct Datum {
        T        total = 0.0;
        unsigned count = 0;
    };

    unsigned num_samples_;
    unsigned time_point_counter_ = 0;

    jblib::Array<double, 2> sz_data_;   // index is spin, time. This order gives a large speedup
};

#endif //JAMS_SPIN_CORRELATION_H
