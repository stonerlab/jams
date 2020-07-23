//
// Created by Joseph Barker on 2019-08-01.
//

#ifndef JAMS_MAGNON_SPECTRUM_H
#define JAMS_MAGNON_SPECTRUM_H

#include <fstream>
#include <complex>
#include <vector>

#include <libconfig.h++>

#include "jams/interface/fft.h"
#include "jams/core/types.h"
#include "jams/core/monitor.h"
#include "jams/monitors/spectrum_base.h"

class Solver;

class MagnonSpectrumMonitor : public SpectrumBaseMonitor {
public:
    using Complex = std::complex<double>;
    using Mat3cx  = std::array<std::array<Complex, 3>, 3>;

    explicit MagnonSpectrumMonitor(const libconfig::Setting &settings);
    ~MagnonSpectrumMonitor() override = default;

    void post_process() override {};
    void update(Solver *solver) override;
    bool is_converged() override { return false; }

private:
    void output_total_magnon_spectrum();
    void output_site_resolved_magnon_spectrum();

    // Toggle outputting spectrum for each site in the unit cell as individual files
    bool do_site_resolved_output_ = false;

    jams::MultiArray<Mat3cx, 3> calculate_magnon_spectrum(const jams::MultiArray<Vec3cx, 3>& spectrum);

    jams::MultiArray<Mat3, 1>   transformations_;
    jams::MultiArray<double, 2> transformed_spins_;
    jams::MultiArray<Mat3cx,3> cumulative_magnon_spectrum_;
};

#endif //JAMS_MAGNON_SPECTRUM_H
