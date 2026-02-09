//
// Created by Joseph Barker on 2019-08-01.
//

#ifndef JAMS_MAGNON_DENSITY_H
#define JAMS_MAGNON_DENSITY_H

#include <jams/core/types.h>
#include <jams/core/monitor.h>
#include <jams/monitors/spectrum_base.h>

class Solver;

class MagnonDensityMonitor : public SpectrumBaseMonitor {
public:
    explicit MagnonDensityMonitor(const libconfig::Setting &settings);
    ~MagnonDensityMonitor() override = default;

    void post_process() override {};
    void update(Solver& solver) override;

private:
    void output_magnon_density();
    void accumulate_magnon_density();

    // cumulative_magnon_density_(frequency_index)
    jams::MultiArray<double,1> cumulative_magnon_density_;
};

#endif //JAMS_MAGNON_DENSITY_H
