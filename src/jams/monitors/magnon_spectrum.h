//
// Created by Joseph Barker on 2019-08-01.
//

#ifndef JAMS_MAGNON_SPECTRUM_H
#define JAMS_MAGNON_SPECTRUM_H

#include <jams/core/types.h>
#include <jams/core/monitor.h>
#include <jams/monitors/spectrum_base.h>
#include <jams/helpers/mixed_precision.h>

#include <fstream>
#include <complex>
#include <vector>

class Solver;

class MagnonSpectrumMonitor : public SpectrumBaseMonitor {
public:
    using Mat3cx  = std::array<std::array<jams::ComplexHi, 3>, 3>;

    explicit MagnonSpectrumMonitor(const libconfig::Setting &settings);
    ~MagnonSpectrumMonitor() override = default;

    void post_process() override {};
    void update(Solver& solver) override;

private:


    void output_total_magnon_spectrum();
    void output_site_resolved_magnon_spectrum();
    void output_magnon_density();

    // Toggle outputting the total magnon spectrum files
    bool do_magnon_spectrum_output_ = true;
    // Toggle calculating and outputting the magnon density
    bool do_magnon_density_ = false;
    // Toggle outputting spectrum for each site in the unit cell as individual files
    bool do_site_resolved_output_ = false;

    void accumulate_magnon_spectrum();



    /// @details cumulative_magnon_spectrum_(motif_index, frequency_index, k_index)[component]
    /// component: 0: +- | 1: -+ | 2: zz
    jams::MultiArray<Vec3,2> cumulative_magnon_spectrum_;
};

#endif //JAMS_MAGNON_SPECTRUM_H
